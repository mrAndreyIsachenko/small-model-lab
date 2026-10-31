# Sprint 1 Experimental Report (Auto-Generated)

> Generated strictly from machine-readable artifacts in `results` (`include_dry_runs=False`).

## Summary Table

| Run ID | Hyp. | Mode | Model | Non-Emb Params | Tokens | Val bpb | Tok/s | MFU | Hardware | Billed Cost ($) | Git SHA |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | :--- |
| `h1_gpt2_parity` | `H1` | `measured` | `gpt2-124m` | `85,842,432` | `128` | `n/a` | `30` | `n/a` | `apple-m4-pro-48gb` | `$0.0000` | `f122a0c9` |

## Run Conclusions

### `h1_gpt2_parity` (`H1` — `measured`)
- **Prediction:** Max absolute logit difference < 1.0e-4 in fp32 across random and natural prompts; future-token and cross-document gradient norm == 0.0.
- **Conclusion:** H1 verified on full 124,439,808-parameter GPT-2 124M: max_abs_logit_diff=0.00e+00 (< 1e-4), sdpa_vs_naive=5.96e-06, kv_cache_diff=5.13e-06, cross_doc_leakage=0.00e+00.
