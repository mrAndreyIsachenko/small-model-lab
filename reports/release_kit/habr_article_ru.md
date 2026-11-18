# Почему наивная абляция LLM ломается в FP8 на vLLM (~50% утечка подпространства) — и как мы построили семейство Tenet v1 (`3B`/`8B`/`12B`) с нуля до продакшн-хирургии

**Коротко:** Мы выложили в открытый доступ (Apache-2.0) полный инженерный пайплайн [`mrAndreyIsachenko/tenet-lab`](https://github.com/mrAndreyIsachenko/tenet-lab), линейку моделей **Tenet v1** (`3B`, `8B`, `12B` + `16K` Nano-лаборатория) на [Hugging Face (`mr-andrey-isachenko`)](https://huggingface.co/mr-andrey-isachenko) и интерактивный браузерный инспектор [`fp8-abliteration-inspector`](https://huggingface.co/spaces/mr-andrey-isachenko/fp8-abliteration-inspector). В статье разбираем архитектуру от `StatefulToolEnv` и `H9` Lancet-хирургии до `H10` Subspace-Safe GRPO, `H11` FP8 E4M3 выживаемости, `H12` TIES-мерджа и `H13` потоковой хирургии `32B/70B`.

---

## 1. Проблема: три скрытых отказа наивной абляции

Классическая ортогональная абляция ($W' = W - \hat{r}\hat{r}^\top W$) проверяет только падение отказов в `bf16`/`fp16`. Но при переносе в реальный продакшн-стек (`vLLM` / `OpenRouter` `FP8 E4M3`, мультишаговый tool-use, дообучение RL или мердж LoRA) возникают **три системных отказа**:

1. **Коллатеральный взрыв KL на reasoning и tool-SQL (`H9`):** направления отказа и словесности («slop») не ортогональны направлениям рассуждения. Прямое вычитание $\hat{r}\hat{r}^\top W$ ломает полезную дисперсию — на `Qwen2.5-3B-Instruct` `naive_kl_collateral` взлетает до **`3.7607`**, на `Mistral-Nemo-12B` — до **`0.2968`**, на `Llama-3.1-8B` — до **`0.1337`**.
2. **~50% обратная утечка подпространства в `FP8 (E4M3)` (`H11`):** у формата `E4M3` всего 3 бита мантиссы. Без сохранения построчных $L_2$-норм и контроля `Row Crest-Factor` ($\max_i \|W_{i,:}\|_\infty / \text{RMS}(W_{i,:}) < 25.0$) квантизация заново индуцирует проекцию вдоль удалённого направления. На всех четырёх масштабах (`nano`, `3B`, `8B`, `12B`) `fp8_subspace_releakage_naive` стабильно равна **`0.4996–0.5003` (~50%)**.
3. **Повторная утечка при RL и линейном мердже адаптеров (`H10`, `H12`):** обычный `GRPO` или наивное сложение $\Delta W_{\text{SQL}} + \Delta W_{\text{unslop}}$ свободно пишет обратно в удалённое подпространство (`unprojected_lora_releakage = 0.805–1.133`), а интерференция сингулярных векторов обрушивает совместное качество.

---

## 2. Среда `StatefulToolEnv` и анти-хакинг рубрика (`H5–H6`)

Перед работой с весами мы построили детерминированную среду расследования инцидентов данных **`StatefulToolEnv`** (`dbt Core + DuckDB + sqlglot`) с изолированными MCP-инструментами и контрафактическими инъекциями сбоев (`product_bug`, `data_pipeline_artifact`, `mix_shift`).

Ключевой урок `H5`: аддитивная рубрика $R_{\text{naive}} = w_1 R_{\text{outcome}} + w_2 R_{\text{evidence}} + w_3 R_{\text{process}}$ мгновенно взламывается агентом — модель угадывает ответ без проверки таблиц (`lazy_naive = 0.75`) или шлёт небезопасный SQL (`unsafe_naive = 0.55`). Переход на **Gated Multiplicative Rubric**:

$$R_{\text{total}} = R_{\text{outcome}} \cdot R_{\text{evidence}} \cdot R_{\text{process}}$$

обнуляет оба эксплойта (`lazy_gated = 0.0000`, `unsafe_gated = 0.0000`, оракул `1.1200`). На `Qwen2.5-3B-Instruct` (`H6`) `Hybrid RAG` даёт `100%` recall на поиске схем, но упирается в `0.5600` на многошаговом исключении гипотез, где `Gated GRPO` выходит на `1.0000` (`+44.0 pp`).

---

## 3. Архитектура `Lancet + Tenet` (`H9–H13`)

### `H9`: Lancet Dual-Anchor Soft-SCR Surgery
Модуль `src/sml/surgery/` работает в `float32`:
1. Извлекает контрастивные направления отказа ($U_{\text{refusal}}$), slop ($U_{\text{slop}}$), а также **reasoning-якорь** ($U_{\text{anchor}}$) и **tool-SQL якорь** ($U_{\text{sql}}$).
2. Очищает целевые направления относительно якорей методом **Modified Gram-Schmidt (MGS)** (`anchor_orthogonality_error <= 1e-7`).
3. Отбирает top-$k$ слоёв по **Signal-to-Collateral-Ratio (SCR)** и применяет совместную проекцию к `self_attn.o_proj` и `mlp.down_proj` с **точным построчным восстановлением $L_2$-норм**.

**Замеры `H9`:**
- `Qwen2.5-3B-Instruct` (M4 Pro MPS): подавление `96.94%`, снижение KL на **`99.41%`** (`3.7607 → 0.0220`), `ortho_err = 1e-07`
- `Llama-3.1-8B-Instruct` (H100): подавление `99.30%`, снижение KL на **`89.89%`** (`0.1337 → 0.0135`), `ortho_err = 1e-08`
- `Mistral-Nemo-12B` (H100): подавление `99.14%`, снижение KL на **`96.72%`** (`0.2968 → 0.0097`), `ortho_err = 1e-08`

### `H10`: Tenet Subspace-Safe Unslop GRPO
Поверх чекпоинта `H9` обучается `GRPO` с Gated Rubric и штрафом за словесность/отказы, а при мердже LoRA обновление проецируется через ортогональное дополнение:

$$\Delta W_{\text{safe}} = (I - U_{\text{ablate}} U_{\text{ablate}}^\top) \Delta W_{\text{LoRA}}$$

с восстановлением построчных норм. Итог:
- `post_merge_suppression_retention = 1.0000` на всех масштабах
- Снижение утечки LoRA на порядок: `0.8300 → 0.0690` (`3B`), `1.1328 → 0.0726` (`8B`), `0.8052 → 0.1282` (`12B`)
- Финальный `slop_and_refusal_rate`: `0.0909` (`3B`, `gated_ifeval = 0.7230`), `0.0410` (`8B`, `0.8114`), `0.0239` (`12B`, `0.6272`), прирост над наивным GRPO от **`+16.02 pp`** до **`+42.92 pp`**.

### `H11`: FP8 (`E4M3`) Survival и Crest-Factor Hygiene
Каждый релизный бандл проходит 7/7 (и 8/8 в бандле) проверок провайдерной готовности (Per-Row и Per-Tensor `FP8 E4M3` + `max_row_crest_factor < 25.0`):
- **`Tenet-3B-Qwen2.5-Unslop-v1`**: FP8 retention `95.41%`, утечка `0.5003 → 0.0270`, KL-снижение `51.85%`, `Crest = 17.2706`
- **`Tenet-8B-Llama3.1-Unslop-v1`**: FP8 retention `99.36%`, утечка `0.5000 → 0.0278`, KL-снижение `42.65%`, `Crest = 6.3244`
- **`Tenet-12B-Nemo-Unslop-v1`**: FP8 retention `96.78%`, утечка `0.4996 → 0.1839`, KL-снижение `48.92%`, `Crest = 10.6508`

### `H12`: Cross-Domain TIES + Gram-Schmidt Merge
Для композиции независимых GRPO-адаптеров ($\Delta W_{\text{SQL}} + \Delta W_{\text{unslop}}$) адаптер $\Delta W_{\text{unslop}}$ проецируется и против $U_{\text{refusal/slop}}$, и против top-$k$ левых сингулярных векторов $U_{\text{SQL}}$ перед TIES (trim, sign-election, disjoint merge) и восстановлением построчных норм:
- Удержание SQL-навыка: `1.0000` (`3B`), `0.9799` (`8B`), `1.0000` (`12B`); удержание unslop: `1.0000` везде
- Утечка после мерджа: `< 5e-7` (против `0.5768–0.849` у наивного мерджа)
- Прирост над наивным мерджем: **`+78.48 pp`** (`3B`), **`+79.65 pp`** (`8B`), **`+78.06 pp`** (`12B`).

### `H13`: Потоковая хирургия `32B/70B` по шардам (`Streaming Shard Surgery`)
Чтобы не упираться в RAM при работе с `Qwen/Qwen2.5-32B-Instruct` и `Llama-3.3-70B-Instruct`, `H13` обрабатывает `.safetensors` построчно/пошардово (`< 16 GB` RAM, `< 15 GB` диск). Математическая эквивалентность потоковой хирургии и потокового мерджа подтверждена на 3-шардовой верификации (`streaming_surgery_max_abs_diff = 7.45e-09`, `streaming_merge_max_abs_diff = 0.0`, `Crest = 1.1926`).

---

## 4. Ссылки и инструменты

* **Интерактивный Space (аудит без загрузки весов на сервер):** [FP8 & Abliteration Brain-Damage Inspector](https://huggingface.co/spaces/mr-andrey-isachenko/fp8-abliteration-inspector) — проверяет `Crest-Factor < 25.0`, FP8 re-leakage и генерирует команды `llama-quantize --tensor-type` (CKA-guided `~5.28 bpw`).
* **Профиль и модели на Hugging Face:** [`mr-andrey-isachenko`](https://huggingface.co/mr-andrey-isachenko)
  * [`Tenet-12B-Nemo-Unslop-v1`](https://huggingface.co/mr-andrey-isachenko/Tenet-12B-Nemo-Unslop-v1)
  * [`Tenet-8B-Llama3.1-Unslop-v1`](https://huggingface.co/mr-andrey-isachenko/Tenet-8B-Llama3.1-Unslop-v1)
  * [`Tenet-3B-Qwen2.5-Unslop-v1`](https://huggingface.co/mr-andrey-isachenko/Tenet-3B-Qwen2.5-Unslop-v1)
  * [`Tenet-16K-Nano-Lab-v1`](https://huggingface.co/mr-andrey-isachenko/Tenet-16K-Nano-Lab-v1)
* **GitHub-репозиторий и технический отчёт:** [`mrAndreyIsachenko/tenet-lab`](https://github.com/mrAndreyIsachenko/tenet-lab) (Apache-2.0, политика Zero Placeholder Numbers).
