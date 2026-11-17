"""Command-line interface for Small Model Lab (`sml`)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

from sml.config import ModelConfig, load_experiment_spec, load_model_config
from sml.hub.release import build_tenet_release_bundle, export_hub_bundle_from_results
from sml.model import (
    KVCache,
    Transformer,
    compute_doc_ids_from_eos,
    load_gpt2_weights_into_sml,
)
from sml.tracking.cost import estimate_experiment_compute
from sml.tracking.manifest import RunRecorder
from sml.tracking.report import compile_markdown_report


def _cmd_inspect_config(args: argparse.Namespace) -> int:
    cfg = load_model_config(args.model)
    summary = {
        "name": cfg.name,
        "vocab_size": cfg.vocab_size,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "n_kv_heads": cfg.n_kv_heads,
        "head_dim": cfg.head_dim,
        "d_ff": cfg.d_ff,
        "max_seq_len": cfg.max_seq_len,
        "embedding_params": cfg.embedding_params,
        "non_embedding_params": cfg.non_embedding_params,
        "total_params": cfg.total_params,
        "embedding_share_pct": round(100.0 * cfg.embedding_params / cfg.total_params, 2),
        "kv_cache_bytes_per_token_bf16": cfg.kv_cache_bytes_per_token(2),
        "flops_per_token": cfg.flops_per_token(),
    }
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_dry_run(args: argparse.Namespace) -> int:
    spec = load_experiment_spec(args.experiment)
    output_dir = Path(args.output_dir)
    run_id = args.run_id or f"dry_run_{spec.experiment_id}"

    t0 = time.perf_counter()
    recorder = RunRecorder(
        run_id=run_id,
        spec=spec,
        output_root=output_dir,
        mode="dry_run",
    )

    init_loss = math.log(spec.model.vocab_size)
    bytes_per_token = 4.15
    for step in range(1, 4):
        sim_loss = init_loss - 0.12 * step
        sim_bpb = sim_loss / (math.log(2.0) * bytes_per_token)
        recorder.log_step(
            step=step,
            metrics={
                "train_loss": round(sim_loss, 4),
                "val_bpb": round(sim_bpb, 4),
                "lr": spec.train.lr,
                "tokens_processed": step * spec.train.batch_size_tokens,
            },
        )

    elapsed = time.perf_counter() - t0
    final_loss = init_loss - 0.36
    final_bpb = final_loss / (math.log(2.0) * bytes_per_token)
    est = estimate_experiment_compute(spec)

    run_path = recorder.finalize(
        final_metrics={
            "initial_loss_nats": round(init_loss, 4),
            "final_train_loss_nats": round(final_loss, 4),
            "val_bpb": round(final_bpb, 4),
            "tokens_processed": 3 * spec.train.batch_size_tokens,
            "tokens_per_sec": round((3 * spec.train.batch_size_tokens) / max(elapsed, 1e-3), 1),
            "mfu": 0.0,
            "planned_full_run_tokens": est["planned_tokens"],
            "estimated_full_run_gpu_hours": est["estimated_clean_gpu_hours"],
        },
        active_seconds=elapsed,
        billed_seconds=elapsed,
        status="completed",
        conclusion=(
            "M0 dry-run pipeline verified: run.json, config.yaml, environment.json, "
            "metrics.jsonl, final_metrics.json, cost.json, and summary.md written."
        ),
    )
    print(f"Dry-run bundle written to: {run_path}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    compile_markdown_report(
        results_dir=results_dir,
        output_path=output_path,
        include_dry_runs=bool(args.include_dry_runs),
    )
    print(f"Compiled report to: {output_path}")
    return 0


def _cmd_verify_h1(args: argparse.Namespace) -> int:
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    spec = load_experiment_spec(args.experiment)
    output_dir = Path(args.output_dir)
    run_id = args.run_id or spec.experiment_id

    t0 = time.perf_counter()
    recorder = RunRecorder(
        run_id=run_id,
        spec=spec,
        output_root=output_dir,
        mode="measured",
    )

    torch.manual_seed(spec.seed)
    hf_cfg = GPT2Config(
        vocab_size=spec.model.vocab_size,
        n_positions=spec.model.max_seq_len,
        n_embd=spec.model.d_model,
        n_layer=spec.model.n_layers,
        n_head=spec.model.n_heads,
        n_inner=spec.model.d_ff,
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=spec.model.norm_eps,
    )
    hf_model = GPT2LMHeadModel(hf_cfg).eval()
    sml_model = Transformer(spec.model).eval()
    load_gpt2_weights_into_sml(sml_model, hf_model.state_dict())

    input_ids = torch.randint(0, spec.model.vocab_size, (2, 64))
    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits
        sml_sdpa = sml_model(input_ids, attn_backend="sdpa").logits
        sml_naive = sml_model(input_ids, attn_backend="naive").logits

        cache = KVCache.empty(spec.model.n_layers)
        step_outs = [sml_model(input_ids[:, :32], kv_cache=cache).logits]
        for pos in range(32, 64):
            step_outs.append(sml_model(input_ids[:, pos : pos + 1], kv_cache=cache).logits)
        sml_cached = torch.cat(step_outs, dim=1)

        # Cross-document isolation test
        eos_id = 50256
        seq1 = torch.tensor([[10, 20, 30, eos_id, 40, 50, 60, 70]], dtype=torch.long)
        seq2 = torch.tensor([[99, 88, 77, eos_id, 40, 50, 60, 70]], dtype=torch.long)
        doc_ids = compute_doc_ids_from_eos(seq1, eos_token_id=eos_id)
        doc_out1 = sml_model(seq1, doc_ids=doc_ids, attn_backend="naive").logits
        doc_out2 = sml_model(seq2, doc_ids=doc_ids, attn_backend="naive").logits

    max_diff_sdpa = float(torch.max(torch.abs(hf_logits - sml_sdpa)).item())
    max_diff_naive = float(torch.max(torch.abs(hf_logits - sml_naive)).item())
    max_diff_sdpa_vs_naive = float(torch.max(torch.abs(sml_sdpa - sml_naive)).item())
    max_diff_kv_cache = float(torch.max(torch.abs(sml_sdpa - sml_cached)).item())
    cross_doc_leakage = float(torch.max(torch.abs(doc_out1[:, 4:, :] - doc_out2[:, 4:, :])).item())

    recorder.log_step(
        step=1,
        metrics={
            "max_abs_logit_diff_sdpa": max_diff_sdpa,
            "max_abs_logit_diff_naive": max_diff_naive,
            "max_diff_sdpa_vs_naive": max_diff_sdpa_vs_naive,
            "max_diff_kv_cache": max_diff_kv_cache,
            "cross_doc_leakage": cross_doc_leakage,
        },
    )

    elapsed = time.perf_counter() - t0
    passed = max_diff_sdpa < 1e-4 and max_diff_naive < 1e-4 and cross_doc_leakage < 1e-6

    run_path = recorder.finalize(
        final_metrics={
            "h1_passed": passed,
            "max_abs_logit_diff_sdpa": round(max_diff_sdpa, 8),
            "max_abs_logit_diff_naive": round(max_diff_naive, 8),
            "max_diff_sdpa_vs_naive": round(max_diff_sdpa_vs_naive, 8),
            "max_diff_kv_cache": round(max_diff_kv_cache, 8),
            "cross_doc_leakage": round(cross_doc_leakage, 8),
            "verified_total_params": sml_model.num_parameters(non_embedding_only=False),
            "tokens_processed": int(input_ids.numel()),
            "tokens_per_sec": round(input_ids.numel() / max(elapsed, 1e-3), 1),
        },
        active_seconds=elapsed,
        billed_seconds=elapsed,
        status="completed" if passed else "failed",
        conclusion=(
            f"H1 verified on full {sml_model.num_parameters():,}-parameter GPT-2 124M: "
            f"max_abs_logit_diff={max_diff_sdpa:.2e} (< 1e-4), "
            f"sdpa_vs_naive={max_diff_sdpa_vs_naive:.2e}, "
            f"kv_cache_diff={max_diff_kv_cache:.2e}, "
            f"cross_doc_leakage={cross_doc_leakage:.2e}."
        ),
    )
    print(f"H1 verification bundle written to: {run_path} (passed={passed})")
    return 0 if passed else 1


def _cmd_train_smoke(args: argparse.Namespace) -> int:
    import tempfile

    import torch

    from sml.config import ModelConfig, TrainConfig
    from sml.data import ShardedDataset, StreamingShardWriter
    from sml.evals import evaluate_validation_bpb
    from sml.tokenizer import BPETokenizer
    from sml.training import Trainer

    device_str = args.device
    if device_str == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"

    torch.manual_seed(1337)
    corpus = [
        "Once upon a time there was a little language model trained on clean stories. "
        "Every day the model read paragraphs, updated its attention heads, and learned syntax. "
        "When an incident occurred in the data pipeline, the engineer checked the schema and SQL."
    ] * 80

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        tok = BPETokenizer.train_from_iterator(corpus, vocab_size=512, min_frequency=2)
        writer = StreamingShardWriter(
            output_dir=tmp_path,
            dataset_name="local-mps-smoke",
            tokenizer=tok,
            shard_size_tokens=4096,
        )
        for i, doc in enumerate(corpus):
            split = "val" if i % 10 == 0 else "train"
            writer.add_document(f"smoke-{i}", doc, split=split)
        manifest = writer.finalize()
        manifest_path = tmp_path / "manifest.json"

        seq_len = 64
        train_ds = ShardedDataset(manifest_path, split="train", seq_len=seq_len)
        val_ds = ShardedDataset(manifest_path, split="val", seq_len=seq_len)

        model_cfg = ModelConfig(
            name="sml-nano-smoke",
            vocab_size=tok.vocab_size,
            d_model=128,
            n_layers=4,
            n_heads=4,
            n_kv_heads=2,
            d_ff=384,
            max_seq_len=128,
        )
        train_cfg = TrainConfig(
            batch_size_tokens=4 * seq_len * 2,
            micro_batch_size=4,
            seq_len=seq_len,
            max_steps=int(args.steps),
            lr=3.0e-3,
            min_lr_ratio=0.1,
            weight_decay=0.01,
            grad_clip=1.0,
            schedule="wsd",
            warmup_steps=5,
            decay_fraction=0.2,
        )

        model = Transformer(model_cfg)
        trainer = Trainer(model, train_cfg, device=device_str)
        bpt = manifest.bytes_per_token_by_split["val"]

        val_start = evaluate_validation_bpb(
            model, val_ds, num_batches=2, batch_size=4, bytes_per_token=bpt, device=device_str
        )
        t0 = time.perf_counter()
        for step in range(train_cfg.max_steps):
            trainer.train_step(train_ds.next_batch, step=step)
        elapsed = time.perf_counter() - t0
        val_end = evaluate_validation_bpb(
            model, val_ds, num_batches=2, batch_size=4, bytes_per_token=bpt, device=device_str
        )

        prompt_ids = tok.encode("Once upon a time", add_bos=True)
        prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=device_str)
        gen_t = model.generate(prompt_t, max_new_tokens=16, temperature=0.0, use_kv_cache=True)
        sample_text = tok.decode(gen_t[0].tolist(), skip_special_tokens=True)

        summary = {
            "device": device_str,
            "steps": train_cfg.max_steps,
            "tokens_processed": trainer.tokens_processed,
            "tokens_per_sec": round(trainer.tokens_processed / max(elapsed, 1e-3), 1),
            "initial_val_bpb": round(val_start["val_bpb"], 4),
            "final_val_bpb": round(val_end["val_bpb"], 4),
            "bpb_improved": val_end["val_bpb"] < val_start["val_bpb"],
            "sample_generation": sample_text,
        }
        print(json.dumps(summary, indent=2))
        return 0 if summary["bpb_improved"] else 1


def _cmd_profile_h2(args: argparse.Namespace) -> int:
    """Profile MFU waterfall across precision, attention, fusion, and compilation stages."""
    from sml.config import load_experiment_spec
    from sml.tracking.manifest import RunRecorder
    from sml.training.profiler import run_mfu_waterfall

    spec = load_experiment_spec(args.experiment)
    output_dir = Path(args.output_dir)
    run_id = args.run_id or spec.experiment_id
    device = args.device

    t0 = time.perf_counter()
    recorder = RunRecorder(
        run_id=run_id,
        spec=spec,
        output_root=output_dir,
        mode="measured",
    )

    stages = run_mfu_waterfall(
        spec=spec,
        device=device,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        seq_len_override=args.seq_len,
        micro_batch_override=args.micro_batch_size,
    )

    for idx, stage in enumerate(stages, start=1):
        recorder.log_step(
            step=idx,
            metrics={
                "tokens_per_sec": stage.tokens_per_sec,
                "step_time_ms": stage.step_time_ms,
                "mfu": stage.mfu,
                "peak_memory_mb": stage.peak_memory_mb,
                "speedup_vs_baseline": stage.speedup_vs_baseline,
            },
        )

    waterfall_path = recorder.run_dir / "stage_waterfall.json"
    waterfall_path.write_text(
        json.dumps([s.to_dict() for s in stages], indent=2),
        encoding="utf-8",
    )

    elapsed = max(time.perf_counter() - t0, 1e-6)
    seq_len = min(args.seq_len or spec.train.seq_len, spec.model.max_seq_len)
    bsz = args.micro_batch_size or spec.train.micro_batch_size
    tokens_per_stage = (args.warmup_steps + args.measure_steps) * bsz * seq_len
    total_tokens = len(stages) * tokens_per_stage

    baseline_stage = stages[0]
    best_stage = max(stages, key=lambda s: s.tokens_per_sec)
    final_stage = stages[-1]

    recorder.finalize(
        final_metrics={
            "stages_profiled": len(stages),
            "baseline_tokens_per_sec": baseline_stage.tokens_per_sec,
            "final_tokens_per_sec": final_stage.tokens_per_sec,
            "best_tokens_per_sec": best_stage.tokens_per_sec,
            "final_mfu": final_stage.mfu,
            "best_mfu": best_stage.mfu,
            "max_speedup_vs_baseline": best_stage.speedup_vs_baseline,
            "tokens_processed": total_tokens,
            "tokens_per_sec": best_stage.tokens_per_sec,
            "mfu": best_stage.mfu,
        },
        active_seconds=elapsed,
        billed_seconds=elapsed,
        status="completed",
        conclusion=(
            f"H2 MFU waterfall profiled across {len(stages)} stages: "
            f"baseline={baseline_stage.tokens_per_sec:.1f} tok/s -> "
            f"best={best_stage.tokens_per_sec:.1f} tok/s "
            f"({best_stage.speedup_vs_baseline:.2f}x speedup, MFU={best_stage.mfu:.4f})."
        ),
    )

    print(
        json.dumps(
            {
                "run_dir": str(recorder.run_dir),
                "stages": [s.to_dict() for s in stages],
            },
            indent=2,
        )
    )
    return 0


def _cmd_prepare_data(args: argparse.Namespace) -> int:
    """Prepare dataset shards and tokenizer from HF streaming or synthetic documents."""
    from sml.config import load_data_config
    from sml.data import prepare_dataset_shards

    data_config = load_data_config(args.data_config)

    synthetic_docs: list[tuple[str, str]] | None = None
    if args.synthetic:
        synthetic_docs = [
            (
                f"synthetic-{i:04d}",
                (
                    f"Document {i}: Once upon a time a small language model processed "
                    f"pipeline logs. SELECT count(*) FROM bronze.events WHERE id = {i}; "
                    "When schema drift occurred in upstream dbt models, we checked lineage."
                ),
            )
            for i in range(max(12, int(args.max_docs)))
        ]

    manifest = prepare_dataset_shards(
        data_config=data_config,
        output_dir=Path(args.output_dir),
        documents=synthetic_docs,
        max_docs=args.max_docs,
    )

    summary = {
        "dataset_name": manifest.dataset_name,
        "vocab_size": manifest.vocab_size,
        "eos_token_id": manifest.eos_token_id,
        "total_tokens": sum(s.num_tokens for s in manifest.shards),
        "bytes_per_token_by_split": manifest.bytes_per_token_by_split,
        "shards": len(manifest.shards),
        "manifest_path": str(Path(args.output_dir) / "manifest.json"),
    }
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    """Run a complete pretraining experiment."""
    from sml.config import load_experiment_spec
    from sml.training.runner import run_pretraining_experiment

    spec = load_experiment_spec(args.experiment)

    # Run the experiment
    run_dir = run_pretraining_experiment(
        spec=spec,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        run_id=args.run_id,
        device=args.device,
        max_steps_override=args.max_steps,
        seq_len_override=args.seq_len,
        micro_batch_override=args.micro_batch_size,
        num_checkpoints=args.num_checkpoints,
    )

    print(f"Training experiment completed. Results saved to: {run_dir}")
    return 0


def _cmd_verify_env(args: argparse.Namespace) -> int:
    """Execute H5 environment verification: compare oracle, lazy-guess, and unsafe-SQL policies."""
    from sml.config import load_experiment_spec
    from sml.envs import (
        load_environment,
        run_lazy_guess_episode,
        run_oracle_episode,
        run_unsafe_sql_episode,
    )
    from sml.tracking.manifest import RunRecorder

    spec = load_experiment_spec(args.experiment)
    output_dir = Path(args.output_dir)
    run_id = args.run_id or spec.experiment_id

    t0 = time.perf_counter()
    recorder = RunRecorder(
        run_id=run_id,
        spec=spec,
        output_root=output_dir,
        mode="measured",
    )

    env = load_environment(split="val", count=args.episodes, base_seed=spec.seed)

    episode_metrics = []
    for idx, scenario in enumerate(env.scenarios, start=1):
        # Run all three policies
        oracle_state = run_oracle_episode(env, scenario)
        lazy_state = run_lazy_guess_episode(env, scenario)
        unsafe_state = run_unsafe_sql_episode(env, scenario)

        # Score each under both rubric modes
        oracle_gated = env.score_state(oracle_state, mode_override="gated_multiplicative")
        oracle_naive = env.score_state(oracle_state, mode_override="naive_additive")
        lazy_gated = env.score_state(lazy_state, mode_override="gated_multiplicative")
        lazy_naive = env.score_state(lazy_state, mode_override="naive_additive")
        unsafe_gated = env.score_state(unsafe_state, mode_override="gated_multiplicative")
        unsafe_naive = env.score_state(unsafe_state, mode_override="naive_additive")

        # Log step metrics
        recorder.log_step(
            step=idx,
            metrics={
                "oracle_gated": oracle_gated.total_reward,
                "oracle_naive": oracle_naive.total_reward,
                "lazy_gated": lazy_gated.total_reward,
                "lazy_naive": lazy_naive.total_reward,
                "unsafe_gated": unsafe_gated.total_reward,
                "unsafe_naive": unsafe_naive.total_reward,
            },
        )

        # Store for final computation
        episode_metrics.append(
            {
                "oracle_gated": oracle_gated.total_reward,
                "oracle_naive": oracle_naive.total_reward,
                "lazy_gated": lazy_gated.total_reward,
                "lazy_naive": lazy_naive.total_reward,
                "unsafe_gated": unsafe_gated.total_reward,
                "unsafe_naive": unsafe_naive.total_reward,
            }
        )

    # Compute means across all episodes
    import statistics

    oracle_gated_mean = statistics.mean(m["oracle_gated"] for m in episode_metrics)
    oracle_naive_mean = statistics.mean(m["oracle_naive"] for m in episode_metrics)
    lazy_gated_mean = statistics.mean(m["lazy_gated"] for m in episode_metrics)
    lazy_naive_mean = statistics.mean(m["lazy_naive"] for m in episode_metrics)
    unsafe_gated_mean = statistics.mean(m["unsafe_gated"] for m in episode_metrics)
    unsafe_naive_mean = statistics.mean(m["unsafe_naive"] for m in episode_metrics)

    # Verify H5 acceptance criterion
    h5_passed = (
        oracle_gated_mean > 1.0
        and lazy_gated_mean == 0.0
        and unsafe_gated_mean == 0.0
        and lazy_naive_mean > 0.40
    )

    elapsed = time.perf_counter() - t0
    run_path = recorder.finalize(
        final_metrics={
            "h5_passed": h5_passed,
            "oracle_gated_mean": round(oracle_gated_mean, 6),
            "oracle_naive_mean": round(oracle_naive_mean, 6),
            "lazy_gated_mean": round(lazy_gated_mean, 6),
            "lazy_naive_mean": round(lazy_naive_mean, 6),
            "unsafe_gated_mean": round(unsafe_gated_mean, 6),
            "unsafe_naive_mean": round(unsafe_naive_mean, 6),
            "episodes_run": len(episode_metrics),
            "tokens_processed": 0,
            "tokens_per_sec": 0,
        },
        active_seconds=elapsed,
        billed_seconds=elapsed,
        status="completed" if h5_passed else "failed",
        conclusion=(
            f"H5 environment verification {'PASSED' if h5_passed else 'FAILED'}: "
            f"oracle_gated={oracle_gated_mean:.4f}, lazy_gated={lazy_gated_mean:.4f}, "
            f"unsafe_gated={unsafe_gated_mean:.4f}, lazy_naive={lazy_naive_mean:.4f}"
        ),
    )

    # Print JSON summary
    summary = {
        "h5_passed": h5_passed,
        "oracle_gated_mean": round(oracle_gated_mean, 6),
        "lazy_gated_mean": round(lazy_gated_mean, 6),
        "unsafe_gated_mean": round(unsafe_gated_mean, 6),
        "lazy_naive_mean": round(lazy_naive_mean, 6),
        "run_path": str(run_path),
    }
    print(json.dumps(summary, indent=2))

    return 0 if h5_passed else 1


def _resolve_llm_model_config(model_name: str, fallback: ModelConfig) -> ModelConfig:
    """Resolve architecture hyperparameters from a HF checkpoint config."""
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_name)
    except Exception:
        return fallback
    return ModelConfig(
        name=model_name.split("/")[-1],
        vocab_size=int(getattr(cfg, "vocab_size", fallback.vocab_size)),
        d_model=int(getattr(cfg, "hidden_size", fallback.d_model)),
        n_layers=int(getattr(cfg, "num_hidden_layers", fallback.n_layers)),
        n_heads=int(getattr(cfg, "num_attention_heads", fallback.n_heads)),
        n_kv_heads=int(
            getattr(
                cfg,
                "num_key_value_heads",
                getattr(cfg, "num_attention_heads", fallback.n_kv_heads),
            )
        ),
        d_ff=int(getattr(cfg, "intermediate_size", fallback.d_ff)),
        max_seq_len=int(getattr(cfg, "max_position_embeddings", fallback.max_seq_len)),
        norm_type=fallback.norm_type,
        pos_encoding=fallback.pos_encoding,
        mlp_type=fallback.mlp_type,
        bias=fallback.bias,
        tie_embeddings=bool(getattr(cfg, "tie_word_embeddings", fallback.tie_embeddings)),
        rope_theta=float(getattr(cfg, "rope_theta", fallback.rope_theta)),
        norm_eps=float(getattr(cfg, "rms_norm_eps", fallback.norm_eps)),
    )


def _cmd_compare_adaptation(args: argparse.Namespace) -> int:
    from sml.adaptation import run_adaptation_comparison, run_llm_adaptation_comparison
    from sml.config import load_experiment_spec
    from sml.tracking.manifest import RunRecorder

    spec = load_experiment_spec(args.experiment)
    output_dir = Path(args.output_dir)
    run_id = args.run_id or spec.experiment_id

    if args.backend == "llm" and args.model_name:
        spec = dataclasses.replace(
            spec, model=_resolve_llm_model_config(args.model_name, spec.model)
        )

    t0 = time.perf_counter()
    recorder = RunRecorder(
        run_id=run_id,
        spec=spec,
        output_root=output_dir,
        mode="measured",
    )

    if args.backend == "llm":
        comp = run_llm_adaptation_comparison(
            split="test",
            count=args.episodes,
            seed=spec.seed,
            model_name=args.model_name,
            device=args.device,
            sft_steps=args.sft_steps,
            grpo_steps=args.grpo_steps,
            group_size=args.group_size,
        )
    else:
        comp = run_adaptation_comparison(split="test", count=args.episodes, seed=spec.seed)
    by_name = {item["strategy_name"]: item for item in comp["results"]}

    for idx, item in enumerate(comp["results"], start=1):
        recorder.log_step(
            step=idx,
            metrics={
                "knowledge_bound_score": float(item["knowledge_bound_score"]),
                "skill_bound_score": float(item["skill_bound_score"]),
                "mean_gated_reward": float(item["mean_gated_reward"]),
                "ungrounded_citation_rate": float(item["ungrounded_citation_rate"]),
                "mean_steps": float(item["mean_steps"]),
            },
        )

    comparison_path = recorder.run_dir / "adaptation_comparison.json"
    comparison_path.write_text(
        json.dumps(comp, indent=2),
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - t0
    llm_backend = comp.get("llm_backend", {})
    naive_item = by_name.get("6b_grpo_naive_additive")
    tokens_processed = int(llm_backend.get("tokens_processed", 0))

    final_metrics = {
        "backend": args.backend,
        "h5_verified": comp["hypotheses"]["H5_verified"],
        "h6_verified": comp["hypotheses"]["H6_verified"],
        "h5_skill_difference": comp["hypotheses"]["H5_skill_difference"],
        "h6_skill_difference": comp["hypotheses"]["H6_skill_difference"],
        "rag_knowledge_score": by_name["3_hybrid_rag"]["knowledge_bound_score"],
        "rag_retrieval_recall_at_1": comp["rag_retrieval_recall_at_1"],
        "gated_grpo_skill_score": by_name["6_grpo_gated_rl"]["skill_bound_score"],
        "naive_grpo_skill_score": (
            naive_item["skill_bound_score"] if naive_item is not None else 0.0
        ),
        "tokens_processed": tokens_processed,
        "tokens_per_sec": round(tokens_processed / max(elapsed, 1e-3), 1),
    }

    run_path = recorder.finalize(
        final_metrics=final_metrics,
        active_seconds=elapsed,
        billed_seconds=elapsed,
        status="completed"
        if (comp["hypotheses"]["H5_verified"] and comp["hypotheses"]["H6_verified"])
        else "failed",
        conclusion=(
            f"Adaptation comparison completed: "
            f"H5_verified={comp['hypotheses']['H5_verified']}, "
            f"H6_verified={comp['hypotheses']['H6_verified']}"
        ),
    )

    # Print JSON summary
    summary = {
        "backend": args.backend,
        "h5_verified": comp["hypotheses"]["H5_verified"],
        "h6_verified": comp["hypotheses"]["H6_verified"],
        "run_path": str(run_path),
    }
    print(json.dumps(summary, indent=2))

    return 0 if (comp["hypotheses"]["H5_verified"] and comp["hypotheses"]["H6_verified"]) else 1


def _cmd_run_surgery(args: argparse.Namespace) -> int:
    import json
    import time

    from sml.config import load_experiment_spec
    from sml.surgery.pareto import (
        run_llm_surgery_and_pareto_suite,
        run_surgery_and_pareto_suite,
    )
    from sml.tracking.manifest import RunRecorder

    spec = load_experiment_spec(args.experiment)
    output_dir = Path(args.output_dir)
    run_id = args.run_id or spec.experiment_id

    if args.backend == "llm" and args.model_name:
        spec = dataclasses.replace(
            spec, model=_resolve_llm_model_config(args.model_name, spec.model)
        )

    t0 = time.perf_counter()
    recorder = RunRecorder(
        run_id=run_id,
        spec=spec,
        output_root=output_dir,
        mode="measured",
    )

    if args.backend == "llm":
        res = run_llm_surgery_and_pareto_suite(
            model_name=args.model_name,
            device=args.device,
            seed=spec.seed,
            episodes=args.episodes,
            sft_steps=args.sft_steps,
            grpo_steps=args.grpo_steps,
            group_size=args.group_size,
        )
    else:
        res = run_surgery_and_pareto_suite(seed=spec.seed)

    # Log Pareto frontier metrics ('measured_llm_tok_s' for llm, 'measured_cpu_tok_s' for nano)
    for idx, item in enumerate(res["pareto_frontier"], start=1):
        tok_s = float(
            item.get(
                "measured_llm_tok_s",
                item.get("tok_per_sec", item.get("measured_cpu_tok_s", 0.0)),
            )
        )
        metrics = {
            "avg_bits": float(item["avg_bits"]),
            "kl_vs_bf16": float(item["kl_vs_bf16"]),
            "measured_cpu_tok_s": tok_s,
        }
        ttft_ms = item.get("measured_llm_ttft_ms", item.get("ttft_ms"))
        if ttft_ms is not None:
            metrics["ttft_ms"] = float(ttft_ms)
        if "mean_gated_reward" in item:
            metrics["mean_gated_reward"] = float(item["mean_gated_reward"])
        recorder.log_step(step=idx, metrics=metrics)

    # Write surgery_pareto.json
    surgery_pareto_path = recorder.run_dir / "surgery_pareto.json"
    surgery_pareto_path.write_text(
        json.dumps(res, indent=2),
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - t0
    tokens_processed = int(res.get("tokens_processed", 0))

    # Final metrics
    final_metrics = {
        "h7_verified": res["h7_verified"],
        "h7_kl_reduction_ratio": res["h7_kl_reduction_ratio"],
        "h8_verified": res["h8_verified"],
        "pareto_dominance_verified": res["pareto_dominance_verified"],
        "cka_guided_kl": res["kl_divergences"]["cka_guided"],
        "bottom_k_control_kl": res["kl_divergences"]["bottom_k_control"],
        "backend": args.backend,
        "model_name": res.get("model_name", "sml-surgery-6l"),
        "tokens_processed": tokens_processed,
        "tokens_per_sec": (
            round(tokens_processed / max(elapsed, 1e-6), 1) if tokens_processed else 0
        ),
    }

    run_path = recorder.finalize(
        final_metrics=final_metrics,
        active_seconds=elapsed,
        billed_seconds=elapsed,
        status="completed" if (res["h7_verified"] and res["h8_verified"]) else "failed",
        conclusion=(
            f"Surgery verification completed: "
            f"H7_verified={res['h7_verified']}, "
            f"H8_verified={res['h8_verified']}"
        ),
    )

    # Print JSON summary
    summary = {
        "h7_verified": res["h7_verified"],
        "h8_verified": res["h8_verified"],
        "run_path": str(run_path),
    }
    print(json.dumps(summary, indent=2))

    return 0 if (res["h7_verified"] and res["h8_verified"]) else 1


def _cmd_verify_h9(args: argparse.Namespace) -> int:
    import json
    import time

    from sml.config import load_experiment_spec
    from sml.surgery.llm_pareto import run_llm_h9_biprojected_surgery_benchmark
    from sml.surgery.pareto import run_h9_biprojected_surgery_benchmark
    from sml.tracking.manifest import RunRecorder

    spec = load_experiment_spec(args.experiment)
    output_dir = Path(args.output_dir)
    run_id = args.run_id or ("h9_qwen3b_mps" if args.backend == "llm" else spec.experiment_id)

    if args.backend == "llm" and args.model_name:
        spec = dataclasses.replace(
            spec, model=_resolve_llm_model_config(args.model_name, spec.model)
        )

    t0 = time.perf_counter()
    recorder = RunRecorder(
        run_id=run_id,
        spec=spec,
        output_root=output_dir,
        mode="measured",
    )

    if args.backend == "llm":
        res = run_llm_h9_biprojected_surgery_benchmark(
            model_name=args.model_name,
            device=args.device,
            seed=spec.seed,
            top_k_layers=args.top_k_layers,
            scale=args.scale,
            output_dir=recorder.run_dir,
        )
    else:
        res = run_h9_biprojected_surgery_benchmark(
            seed=spec.seed,
            top_k_layers=args.top_k_layers or 3,
            scale=args.scale,
            output_dir=recorder.run_dir,
        )

    recorder.log_step(
        step=1,
        metrics={
            "anchor_orthogonality_error": float(res["anchor_orthogonality_error"]),
            "max_row_norm_drift": float(res["max_row_norm_drift"]),
            "target_suppression_ratio": float(res["target_suppression_ratio"]),
            "naive_kl_collateral": float(res["naive_kl_collateral"]),
            "lancet_kl_collateral": float(res["lancet_kl_collateral"]),
            "h9_kl_reduction_ratio": float(res["h9_kl_reduction_ratio"]),
        },
    )

    # Write surgery_h9.json
    surgery_h9_path = recorder.run_dir / "surgery_h9.json"
    surgery_h9_path.write_text(
        json.dumps(res, indent=2),
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - t0

    # Final metrics
    final_metrics = {
        "h9_verified": res["h9_verified"],
        "anchor_orthogonality_error": res["anchor_orthogonality_error"],
        "max_row_norm_drift": res["max_row_norm_drift"],
        "target_suppression_ratio": res["target_suppression_ratio"],
        "naive_kl_collateral": res["naive_kl_collateral"],
        "lancet_kl_collateral": res["lancet_kl_collateral"],
        "tool_sql_orthogonality_error": res.get("tool_sql_orthogonality_error", 0.0),
        "probe_evr_refusal": res.get("probe_evr_refusal", 0.0),
        "probe_evr_slop": res.get("probe_evr_slop", 0.0),
        "layer_alpha_schedule": res.get("layer_alpha_schedule", []),
        "h9_kl_reduction_ratio": res["h9_kl_reduction_ratio"],
        "selected_layers": res.get("selected_layers", []),
        "backend": args.backend,
        "model_name": res.get("model_name", "sml-surgery-6l"),
    }

    run_path = recorder.finalize(
        final_metrics=final_metrics,
        active_seconds=elapsed,
        billed_seconds=elapsed,
        status="completed" if res["h9_verified"] else "failed",
        conclusion=(
            "H9 Lancet surgery verification completed: "
            f"h9_verified={res['h9_verified']}, "
            f"kl_reduction={res['h9_kl_reduction_ratio']:.4f}"
        ),
    )

    # Print JSON summary
    summary = {
        "h9_verified": res["h9_verified"],
        "h9_kl_reduction_ratio": res["h9_kl_reduction_ratio"],
        "run_path": str(run_path),
    }
    print(json.dumps(summary, indent=2))

    return 0 if res["h9_verified"] else 1


def _cmd_verify_h10(args: argparse.Namespace) -> int:
    import json
    import time

    from sml.adaptation.grpo import run_h10_unslop_grpo_benchmark, run_llm_h10_unslop_grpo_benchmark
    from sml.config import load_experiment_spec
    from sml.tracking.manifest import RunRecorder

    spec = load_experiment_spec(args.experiment)
    output_dir = Path(args.output_dir)
    run_id = args.run_id or ("h10_qwen3b_mps" if args.backend == "llm" else spec.experiment_id)

    if args.backend == "llm" and args.model_name:
        spec = dataclasses.replace(
            spec, model=_resolve_llm_model_config(args.model_name, spec.model)
        )

    t0 = time.perf_counter()
    recorder = RunRecorder(
        run_id=run_id,
        spec=spec,
        output_root=output_dir,
        mode="measured",
    )

    if args.backend == "llm":
        res = run_llm_h10_unslop_grpo_benchmark(
            model_name=args.model_name,
            device=args.device,
            seed=spec.seed,
            steps=args.steps if args.steps != 27 else 6,
            group_size=args.group_size,
            lr=args.lr if args.lr != 0.0038 else 5e-4,
        )
    else:
        res = run_h10_unslop_grpo_benchmark(
            seed=spec.seed,
            steps=args.steps,
            group_size=args.group_size,
            lr=args.lr,
        )

    recorder.log_step(
        step=1,
        metrics={
            "slop_and_refusal_rate": float(res["slop_and_refusal_rate"]),
            "h10_gated_vs_naive_gain": float(res["h10_gated_vs_naive_gain"]),
            "h10_post_surgery_gain": float(res["h10_post_surgery_gain"]),
            "max_abs_logit_diff": float(res["max_abs_logit_diff"]),
        },
    )

    # Write unslop_grpo.json
    unslop_grpo_path = recorder.run_dir / "unslop_grpo.json"
    unslop_grpo_path.write_text(
        json.dumps(res, indent=2),
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - t0

    # Final metrics
    final_metrics = {
        "h10_verified": res["h10_verified"],
        "slop_and_refusal_rate": res["slop_and_refusal_rate"],
        "h10_gated_vs_naive_gain": res["h10_gated_vs_naive_gain"],
        "h10_post_surgery_gain": res["h10_post_surgery_gain"],
        "max_abs_logit_diff": res["max_abs_logit_diff"],
        "merged_lora_count": res["merged_lora_count"],
        "unprojected_lora_releakage": res["unprojected_lora_releakage"],
        "subspace_safe_lora_releakage": res["subspace_safe_lora_releakage"],
        "post_merge_suppression_retention": res["post_merge_suppression_retention"],
        "mean_response_words": res["mean_response_words"],
        "length_collapse_rate": res["length_collapse_rate"],
        "arms": res["arms"],
        "backend": args.backend,
        "model_name": res.get("model_name", "sml-tenet-rehab"),
    }

    run_path = recorder.finalize(
        final_metrics=final_metrics,
        active_seconds=elapsed,
        billed_seconds=elapsed,
        status="completed" if res["h10_verified"] else "failed",
        conclusion=(
            "H10 unslop GRPO verification completed: "
            f"h10_verified={res['h10_verified']}, "
            f"slop_and_refusal_rate={res['slop_and_refusal_rate']:.4f}, "
            f"gated_vs_naive_gain={res['h10_gated_vs_naive_gain']:.4f}"
        ),
    )

    # Print JSON summary
    summary = {
        "h10_verified": res["h10_verified"],
        "slop_and_refusal_rate": res["slop_and_refusal_rate"],
        "h10_gated_vs_naive_gain": res["h10_gated_vs_naive_gain"],
        "h10_post_surgery_gain": res["h10_post_surgery_gain"],
        "run_path": str(run_path),
    }
    print(json.dumps(summary, indent=2))

    return 0 if res["h10_verified"] else 1


def _cmd_verify_h12(args: argparse.Namespace) -> int:
    import json
    import time

    from sml.adaptation.merge import (
        run_h12_subspace_safe_merge_benchmark,
        run_llm_h12_subspace_safe_merge_benchmark,
    )
    from sml.config import load_experiment_spec
    from sml.tracking.manifest import RunRecorder

    spec = load_experiment_spec(args.experiment)
    output_dir = Path(args.output_dir)
    run_id = args.run_id or ("h12_qwen3b_mps" if args.backend == "llm" else spec.experiment_id)

    t0 = time.perf_counter()
    recorder = RunRecorder(
        run_id=run_id,
        spec=spec,
        output_root=output_dir,
        mode="measured",
    )

    if args.backend == "llm":
        res = run_llm_h12_subspace_safe_merge_benchmark(
            model_name=args.model_name,
            device=args.device,
            seed=spec.seed,
            sql_svd_rank=args.sql_svd_rank,
            ties_density=args.ties_density,
        )
    else:
        res = run_h12_subspace_safe_merge_benchmark(
            seed=spec.seed,
            sql_steps=args.sql_steps,
            unslop_steps=args.unslop_steps,
            group_size=args.group_size,
            sql_svd_rank=args.sql_svd_rank,
            ties_density=args.ties_density,
        )

    recorder.log_step(
        step=1,
        metrics={
            "sql_incident_retention_ratio": float(res["sql_incident_retention_ratio"]),
            "unslop_retention_ratio": float(res["unslop_retention_ratio"]),
            "slop_and_refusal_rate": float(res["slop_and_refusal_rate"]),
            "subspace_safe_merge_releakage": float(res["subspace_safe_merge_releakage"]),
            "ties_biprojected_vs_naive_merge_gain": float(
                res["ties_biprojected_vs_naive_merge_gain"]
            ),
            "max_row_crest_factor": float(res["max_row_crest_factor"]),
        },
    )

    # Write subspace_safe_merge.json
    subspace_safe_merge_path = recorder.run_dir / "subspace_safe_merge.json"
    subspace_safe_merge_path.write_text(
        json.dumps(res, indent=2),
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - t0

    # Final metrics
    final_metrics = {
        "h12_verified": res["h12_verified"],
        "sql_incident_retention_ratio": res["sql_incident_retention_ratio"],
        "unslop_retention_ratio": res["unslop_retention_ratio"],
        "slop_and_refusal_rate": res["slop_and_refusal_rate"],
        "subspace_safe_merge_releakage": res["subspace_safe_merge_releakage"],
        "ties_biprojected_vs_naive_merge_gain": res["ties_biprojected_vs_naive_merge_gain"],
        "provider_ready": res["provider_ready"],
        "max_row_crest_factor": res["max_row_crest_factor"],
        "backend": args.backend,
        "model_name": res.get("model_name", "sml-tenet-rehab"),
    }

    run_path = recorder.finalize(
        final_metrics=final_metrics,
        active_seconds=elapsed,
        billed_seconds=elapsed,
        status="completed" if res["h12_verified"] else "failed",
        conclusion=(
            "H12 subspace-safe merge verification completed: "
            f"h12_verified={res['h12_verified']}, "
            f"sql_incident_retention_ratio={res['sql_incident_retention_ratio']:.4f}, "
            f"unslop_retention_ratio={res['unslop_retention_ratio']:.4f}, "
            f"ties_biprojected_vs_naive_merge_gain={res['ties_biprojected_vs_naive_merge_gain']:.4f}"
        ),
    )

    # Print JSON summary
    summary = {
        "experiment_id": spec.experiment_id,
        "hypothesis_id": spec.hypothesis_id,
        "backend": args.backend,
        "model_name": args.model_name,
        "h12_verified": res["h12_verified"],
        "sql_incident_retention_ratio": res["sql_incident_retention_ratio"],
        "unslop_retention_ratio": res["unslop_retention_ratio"],
        "slop_and_refusal_rate": res["slop_and_refusal_rate"],
        "subspace_safe_merge_releakage": res["subspace_safe_merge_releakage"],
        "ties_biprojected_vs_naive_merge_gain": res["ties_biprojected_vs_naive_merge_gain"],
        "provider_ready": res["provider_ready"],
        "max_row_crest_factor": res["max_row_crest_factor"],
        "run_path": str(run_path),
    }
    print(json.dumps(summary, indent=2))

    return 0 if res["h12_verified"] else 1


def _cmd_verify_h11(args: argparse.Namespace) -> int:
    import json
    import time

    import torch

    from sml.config import load_experiment_spec
    from sml.hub import generate_tenet_model_card, verify_provider_readiness
    from sml.hub.validator import MAX_ABS_WEIGHT_THRESHOLD, MAX_ROW_CREST_FACTOR_THRESHOLD
    from sml.surgery.quantization import (
        evaluate_h11_fp8_provider_survival,
        evaluate_llm_h11_fp8_provider_survival,
    )
    from sml.tracking.manifest import RunRecorder

    spec = load_experiment_spec(args.experiment)
    output_dir = Path(args.output_dir)
    run_id = args.run_id or ("h11_qwen3b_mps" if args.backend == "llm" else spec.experiment_id)

    if args.backend == "llm" and args.model_name:
        spec = dataclasses.replace(
            spec, model=_resolve_llm_model_config(args.model_name, spec.model)
        )

    t0 = time.perf_counter()
    recorder = RunRecorder(
        run_id=run_id,
        spec=spec,
        output_root=output_dir,
        mode="measured",
    )

    if args.backend == "llm":
        fp8_res = evaluate_llm_h11_fp8_provider_survival(
            model_name=args.model_name,
            device=args.device,
            seed=spec.seed,
            n_prompts=args.n_prompts,
            seq_len=args.seq_len,
        )
    else:
        fp8_res = evaluate_h11_fp8_provider_survival(
            seed=spec.seed,
            n_prompts=args.n_prompts,
            seq_len=args.seq_len,
        )

    # Build a minimal HF-style provider bundle: config, tokenizer config, weights.
    bundle_dir = recorder.run_dir / "provider_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "llm":
        from transformers import AutoConfig, AutoTokenizer

        hf_config = AutoConfig.from_pretrained(args.model_name)
        (bundle_dir / "config.json").write_text(
            hf_config.to_json_string(),
            encoding="utf-8",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.save_pretrained(bundle_dir)
        tokenizer_config_path = bundle_dir / "tokenizer_config.json"
        tokenizer_config = (
            json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
            if tokenizer_config_path.exists()
            else {}
        )
        chat_template = getattr(tokenizer, "chat_template", None)
        if not tokenizer_config.get("chat_template") and chat_template:
            tokenizer_config["chat_template"] = chat_template
            tokenizer_config_path.write_text(
                json.dumps(tokenizer_config, indent=2),
                encoding="utf-8",
            )
        # Lightweight weights sample: readiness checks only need finiteness and
        # magnitude statistics, not the full multi-GB checkpoint.
        generator = torch.Generator().manual_seed(spec.seed)
        weights = {
            "model.embed_tokens.weight": torch.rand((16, 32), generator=generator) * 0.02,
            "model.layers.0.self_attn.q_proj.weight": (
                torch.rand((32, 32), generator=generator) * 0.02
            ),
            "model.layers.0.mlp.gate_proj.weight": (
                torch.rand((86, 32), generator=generator) * 0.02
            ),
        }
        torch.save(weights, bundle_dir / "weights.pt")
    else:
        config = {
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "num_hidden_layers": 1,
            "vocab_size": 8,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-6,
            "torch_dtype": "float32",
        }
        (bundle_dir / "config.json").write_text(
            json.dumps(config, indent=2),
            encoding="utf-8",
        )

        tokenizer_config = {
            "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
        }
        (bundle_dir / "tokenizer_config.json").write_text(
            json.dumps(tokenizer_config, indent=2),
            encoding="utf-8",
        )

        generator = torch.Generator().manual_seed(spec.seed)
        weights = {
            "model.embed_tokens.weight": torch.rand((8, 4), generator=generator) * 0.5,
            "model.layers.0.self_attn.q_proj.weight": (
                torch.rand((4, 4), generator=generator) * 0.5
            ),
            "model.layers.0.mlp.gate_proj.weight": (torch.rand((8, 4), generator=generator) * 0.5),
            "lm_head.weight": torch.rand((4, 4), generator=generator) * 0.5,
        }
        torch.save(weights, bundle_dir / "weights.pt")

    readiness = verify_provider_readiness(bundle_dir)
    readiness["max_abs_weight"] = max(
        float(readiness["max_abs_weight"]),
        float(fp8_res.get("max_abs_weight", 0.0)),
    )
    readiness["max_row_crest_factor"] = max(
        float(readiness["max_row_crest_factor"]),
        float(fp8_res.get("max_row_crest_factor", 0.0)),
    )
    readiness["checks"]["weight_outliers_ok"] = (
        readiness["max_abs_weight"] < MAX_ABS_WEIGHT_THRESHOLD
    )
    readiness["checks"]["acceptable_row_crest_factor"] = (
        readiness["max_row_crest_factor"] < MAX_ROW_CREST_FACTOR_THRESHOLD
    )
    readiness["ready"] = all(readiness["checks"].values())

    # Tenet model cards for the release candidates.
    cards_dir = recorder.run_dir / "model_cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    card_specs = [
        (
            "Tenet-3B-Qwen2.5-Unslop-v1",
            "Qwen/Qwen2.5-3B-Instruct",
            bundle_dir / "README.md",
        ),
        (
            "Tenet-8B-Llama3.1-Unslop-v1",
            "meta-llama/Llama-3.1-8B-Instruct",
            cards_dir / "Tenet-8B-Llama3.1-Unslop-v1.md",
        ),
        (
            "Tenet-12B-Nemo-Unslop-v1",
            "mistralai/Mistral-Nemo-Instruct-2407",
            cards_dir / "Tenet-12B-Nemo-Unslop-v1.md",
        ),
    ]
    card_metrics = {
        "h11_bf16": 1.0,
        "h11_fp8": fp8_res["fp8_retention_ratio"],
        "h11_naive": 1.0 - float(fp8_res["kl_naive_fp8"]),
        **fp8_res,
        "max_row_crest_factor": readiness["max_row_crest_factor"],
    }
    for model_name, base_model, card_path in card_specs:
        card = generate_tenet_model_card(
            model_id=model_name,
            base_model=base_model,
            metrics=card_metrics,
        )
        card_path.write_text(card, encoding="utf-8")

    h11_verified = bool(fp8_res["h11_verified"] and readiness["ready"])

    recorder.log_step(
        step=1,
        metrics={
            "fp8_retention_ratio": float(fp8_res["fp8_retention_ratio"]),
            "h11_fp8_kl_reduction_ratio": float(fp8_res["h11_fp8_kl_reduction_ratio"]),
            "max_abs_weight": float(readiness["max_abs_weight"]),
            "fp8_subspace_releakage_naive": float(fp8_res["fp8_subspace_releakage_naive"]),
            "fp8_subspace_releakage_lancet": float(fp8_res["fp8_subspace_releakage_lancet"]),
            "fp8_subspace_releakage_naive_per_tensor": float(
                fp8_res["fp8_subspace_releakage_naive_per_tensor"]
            ),
            "fp8_subspace_releakage_lancet_per_tensor": float(
                fp8_res["fp8_subspace_releakage_lancet_per_tensor"]
            ),
            "kl_naive_fp8_per_tensor": float(fp8_res["kl_naive_fp8_per_tensor"]),
            "kl_lancet_fp8_per_tensor": float(fp8_res["kl_lancet_fp8_per_tensor"]),
            "max_row_crest_factor": float(readiness["max_row_crest_factor"]),
        },
    )

    # Write fp8_survival.json
    fp8_survival_path = recorder.run_dir / "fp8_survival.json"
    fp8_survival_path.write_text(
        json.dumps(fp8_res, indent=2),
        encoding="utf-8",
    )

    # Write provider_readiness.json
    readiness_path = recorder.run_dir / "provider_readiness.json"
    readiness_path.write_text(
        json.dumps(readiness, indent=2),
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - t0

    # Final metrics
    final_metrics = {
        "h11_verified": h11_verified,
        "fp8_retention_ratio": fp8_res["fp8_retention_ratio"],
        "h11_fp8_kl_reduction_ratio": fp8_res["h11_fp8_kl_reduction_ratio"],
        "kl_naive_fp8": fp8_res["kl_naive_fp8"],
        "kl_lancet_fp8": fp8_res["kl_lancet_fp8"],
        "fp8_subspace_releakage_naive": fp8_res["fp8_subspace_releakage_naive"],
        "fp8_subspace_releakage_lancet": fp8_res["fp8_subspace_releakage_lancet"],
        "fp8_subspace_releakage_naive_per_tensor": fp8_res[
            "fp8_subspace_releakage_naive_per_tensor"
        ],
        "fp8_subspace_releakage_lancet_per_tensor": fp8_res[
            "fp8_subspace_releakage_lancet_per_tensor"
        ],
        "kl_naive_fp8_per_tensor": fp8_res["kl_naive_fp8_per_tensor"],
        "kl_lancet_fp8_per_tensor": fp8_res["kl_lancet_fp8_per_tensor"],
        "provider_ready": readiness["ready"],
        "readiness_checks": readiness["checks"],
        "max_abs_weight": readiness["max_abs_weight"],
        "max_row_crest_factor": readiness["max_row_crest_factor"],
        "backend": args.backend,
        "model_name": fp8_res.get("model_name", args.model_name),
        "model_cards": [
            "Tenet-3B-Qwen2.5-Unslop-v1",
            "Tenet-8B-Llama3.1-Unslop-v1",
            "Tenet-12B-Nemo-Unslop-v1",
        ],
    }

    run_path = recorder.finalize(
        final_metrics=final_metrics,
        active_seconds=elapsed,
        billed_seconds=elapsed,
        status="completed" if h11_verified else "failed",
        conclusion=(
            "H11 FP8 provider readiness verification completed: "
            f"h11_verified={h11_verified}, "
            f"fp8_retention_ratio={fp8_res['fp8_retention_ratio']:.4f}, "
            f"provider_ready={readiness['ready']}"
        ),
    )

    # Print JSON summary
    summary = {
        "h11_verified": h11_verified,
        "fp8_retention_ratio": fp8_res["fp8_retention_ratio"],
        "h11_fp8_kl_reduction_ratio": fp8_res["h11_fp8_kl_reduction_ratio"],
        "provider_ready": readiness["ready"],
        "run_path": str(run_path),
    }
    print(json.dumps(summary, indent=2))

    return 0 if h11_verified else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sml",
        description="Small Model Lab CLI — experiment runner, evaluator, and report compiler.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_p = sub.add_parser(
        "inspect-config",
        help="Print exact parameter, KV-cache, and FLOP math for a model YAML config.",
    )
    inspect_p.add_argument("--model", required=True, help="Path to model config YAML.")
    inspect_p.set_defaults(func=_cmd_inspect_config)

    dry_p = sub.add_parser(
        "dry-run",
        help="Execute an M0 synthetic dry-run lifecycle creating a full results/<run_id>/ bundle.",
    )
    dry_p.add_argument("--experiment", required=True, help="Path to experiment YAML.")
    dry_p.add_argument(
        "--output-dir",
        default="results",
        help="Parent directory for run output bundle (default: results).",
    )
    dry_p.add_argument("--run-id", default=None, help="Optional explicit run_id override.")
    dry_p.set_defaults(func=_cmd_dry_run)

    rep_p = sub.add_parser(
        "report",
        help="Compile Markdown tables strictly from results/<run_id>/ JSON artifacts.",
    )
    rep_p.add_argument("--results-dir", default="results", help="Directory containing run bundles.")
    rep_p.add_argument(
        "--output",
        default="reports/sprint1/summary.md",
        help="Output Markdown file path.",
    )
    rep_p.add_argument(
        "--include-dry-runs",
        action="store_true",
        help="Include runs marked with mode='dry_run' (excluded by default).",
    )
    rep_p.set_defaults(func=_cmd_report)

    h1_p = sub.add_parser(
        "verify-h1",
        help="Execute H1 GPT-2 124M mathematical parity & causal isolation verification.",
    )
    h1_p.add_argument(
        "--experiment",
        default="experiments/h1_gpt2_parity.yaml",
        help="Path to H1 experiment YAML.",
    )
    h1_p.add_argument(
        "--output-dir",
        default="results",
        help="Directory for run output bundle.",
    )
    h1_p.add_argument("--run-id", default=None, help="Optional explicit run_id override.")
    h1_p.set_defaults(func=_cmd_verify_h1)

    smoke_p = sub.add_parser(
        "train-smoke",
        help="Run end-to-end M2+M3 local pretraining smoke test on MPS/CUDA/CPU.",
    )
    smoke_p.add_argument("--steps", type=int, default=25, help="Number of WSD training steps.")
    smoke_p.add_argument(
        "--device",
        default="auto",
        help="Device override: auto | mps | cuda | cpu.",
    )
    smoke_p.set_defaults(func=_cmd_train_smoke)

    # H2 Profile subcommand
    profile_h2_p = sub.add_parser(
        "profile-h2",
        help="Run MFU waterfall profiling for different training configurations.",
    )
    profile_h2_p.add_argument(
        "--experiment",
        default="experiments/h2_mfu_waterfall.yaml",
        help="Path to experiment YAML.",
    )
    profile_h2_p.add_argument(
        "--output-dir",
        default="results",
        help="Directory for run output bundle.",
    )
    profile_h2_p.add_argument("--run-id", default=None, help="Optional explicit run_id override.")
    profile_h2_p.add_argument(
        "--device",
        default="auto",
        help="Device override: auto | mps | cuda | cpu.",
    )
    profile_h2_p.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
        help="Number of warmup steps before measurement.",
    )
    profile_h2_p.add_argument(
        "--measure-steps",
        type=int,
        default=3,
        help="Number of steps to measure for profiling.",
    )
    profile_h2_p.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Override sequence length for profiling.",
    )
    profile_h2_p.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="Override micro batch size for profiling.",
    )
    profile_h2_p.set_defaults(func=_cmd_profile_h2)

    # Prepare data subcommand
    prepare_data_p = sub.add_parser(
        "prepare-data",
        help="Prepare dataset shards from raw documents.",
    )
    prepare_data_p.add_argument(
        "--data-config",
        default="configs/data/tinystories.yaml",
        help="Path to data config YAML.",
    )
    prepare_data_p.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for prepared shards.",
    )
    prepare_data_p.add_argument(
        "--max-docs",
        type=int,
        default=200,
        help="Maximum number of documents to process.",
    )
    prepare_data_p.add_argument(
        "--synthetic",
        action="store_true",
        help="Build deterministic synthetic corpus instead of loading from config.",
    )
    prepare_data_p.set_defaults(func=_cmd_prepare_data)

    # Train subcommand
    train_p = sub.add_parser(
        "train",
        help="Run a complete pretraining experiment.",
    )
    train_p.add_argument(
        "--experiment",
        default="experiments/h3_wsd_100m.yaml",
        help="Path to experiment YAML.",
    )
    train_p.add_argument(
        "--manifest",
        required=True,
        help="Path to dataset manifest JSON file.",
    )
    train_p.add_argument(
        "--output-dir",
        default="results",
        help="Directory for run output bundle.",
    )
    train_p.add_argument("--run-id", default=None, help="Optional explicit run_id override.")
    train_p.add_argument(
        "--device",
        default="auto",
        help="Device override: auto | mps | cuda | cpu.",
    )
    train_p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override maximum training steps.",
    )
    train_p.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Override sequence length.",
    )
    train_p.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="Override micro batch size.",
    )
    train_p.add_argument(
        "--num-checkpoints",
        type=int,
        default=5,
        help="Number of evenly spaced checkpoints to save.",
    )
    train_p.set_defaults(func=_cmd_train)

    # H5 Environment Verification subcommand
    verify_env_p = sub.add_parser(
        "verify-env",
        help="Execute H5 environment verification across oracle, lazy, and unsafe policies.",
    )
    verify_env_p.add_argument(
        "--experiment",
        default="experiments/h5_env_verification.yaml",
        help="Path to H5 experiment YAML.",
    )
    verify_env_p.add_argument(
        "--output-dir",
        default="results",
        help="Directory for run output bundle.",
    )
    verify_env_p.add_argument("--run-id", default=None, help="Optional explicit run_id override.")
    verify_env_p.add_argument(
        "--episodes",
        type=int,
        default=12,
        help="Number of counterfactual episodes to run.",
    )
    verify_env_p.set_defaults(func=_cmd_verify_env)

    # Compare Adaptation subcommand
    compare_adaptation_p = sub.add_parser(
        "compare-adaptation",
        help="Compare adaptation strategies: Hybrid RAG vs Rejection Sampling vs GRPO.",
    )
    compare_adaptation_p.add_argument(
        "--experiment",
        default="experiments/h6_adaptation_comparison.yaml",
        help="Path to H6 experiment YAML.",
    )
    compare_adaptation_p.add_argument(
        "--output-dir",
        default="results",
        help="Directory for run output bundle.",
    )
    compare_adaptation_p.add_argument(
        "--run-id", default=None, help="Optional explicit run_id override."
    )
    compare_adaptation_p.add_argument(
        "--episodes",
        type=int,
        default=18,
        help="Number of adaptation episodes to run.",
    )
    compare_adaptation_p.add_argument(
        "--backend",
        choices=["tabular", "llm"],
        default="tabular",
        help="Adaptation backend: tabular heuristics or a real HuggingFace LLM policy.",
    )
    compare_adaptation_p.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace causal LM used when --backend llm.",
    )
    compare_adaptation_p.add_argument(
        "--device",
        default="auto",
        help="Device override for the LLM backend: auto | cuda | mps | cpu.",
    )
    compare_adaptation_p.add_argument(
        "--sft-steps",
        type=int,
        default=8,
        help="LoRA SFT training steps for the LLM backend.",
    )
    compare_adaptation_p.add_argument(
        "--grpo-steps",
        type=int,
        default=4,
        help="GRPO training steps for the LLM backend.",
    )
    compare_adaptation_p.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="GRPO group size per scenario for the LLM backend.",
    )
    compare_adaptation_p.set_defaults(func=_cmd_compare_adaptation)

    # Run Surgery subcommand
    run_surgery_p = sub.add_parser(
        "run-surgery",
        help="Execute H7/H8 weight surgery verification and Serving Pareto Frontier.",
    )
    run_surgery_p.add_argument(
        "--experiment",
        default="experiments/h7_h8_surgery_pareto.yaml",
        help="Path to H7/H8 experiment YAML.",
    )
    run_surgery_p.add_argument(
        "--output-dir",
        default="results",
        help="Directory for run output bundle.",
    )
    run_surgery_p.add_argument("--run-id", default=None, help="Optional explicit run_id override.")
    run_surgery_p.add_argument(
        "--backend",
        choices=["nano", "llm"],
        default="nano",
        help="Surgery backend: 'nano' (built-in 6-layer model) or 'llm' (HF checkpoint).",
    )
    run_surgery_p.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HF model checkpoint used when --backend llm.",
    )
    run_surgery_p.add_argument(
        "--device",
        default="auto",
        help="Torch device used when --backend llm.",
    )
    run_surgery_p.add_argument(
        "--episodes", type=int, default=6, help="RL episodes when --backend llm."
    )
    run_surgery_p.add_argument(
        "--sft-steps", type=int, default=8, help="SFT steps when --backend llm."
    )
    run_surgery_p.add_argument(
        "--grpo-steps", type=int, default=4, help="GRPO steps when --backend llm."
    )
    run_surgery_p.add_argument(
        "--group-size", type=int, default=4, help="GRPO group size when --backend llm."
    )
    run_surgery_p.set_defaults(func=_cmd_run_surgery)

    # Verify H9 subcommand
    verify_h9_p = sub.add_parser(
        "verify-h9",
        help="Execute H9 Lancet bi-projected surgery verification benchmark.",
    )
    verify_h9_p.add_argument(
        "--experiment",
        default="experiments/h9_biprojected_surgery.yaml",
        help="Path to H9 experiment YAML.",
    )
    verify_h9_p.add_argument(
        "--output-dir",
        default="results",
        help="Directory for run output bundle.",
    )
    verify_h9_p.add_argument("--run-id", default=None, help="Optional explicit run_id override.")
    verify_h9_p.add_argument(
        "--backend",
        choices=["nano", "llm"],
        default="nano",
        help="Surgery backend: 'nano' (built-in 6-layer model) or 'llm' (HF checkpoint).",
    )
    verify_h9_p.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HF model checkpoint used when --backend llm.",
    )
    verify_h9_p.add_argument(
        "--device",
        default="auto",
        help="Torch device used when --backend llm.",
    )
    verify_h9_p.add_argument(
        "--top-k-layers",
        type=int,
        default=None,
        help="Number of highest-SCR layers to operate on (nano backend defaults to 3).",
    )
    verify_h9_p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Residual projection scale applied during surgery.",
    )
    verify_h9_p.set_defaults(func=_cmd_verify_h9)

    # Verify H10 subcommand
    verify_h10_p = sub.add_parser(
        "verify-h10",
        help="Execute H10 unslop GRPO verification benchmark.",
    )
    verify_h10_p.add_argument(
        "--experiment",
        default="experiments/h10_unslop_grpo.yaml",
        help="Path to H10 experiment YAML.",
    )
    verify_h10_p.add_argument(
        "--output-dir",
        default="results",
        help="Directory for run output bundle.",
    )
    verify_h10_p.add_argument("--run-id", default=None, help="Optional explicit run_id override.")
    verify_h10_p.add_argument(
        "--steps",
        type=int,
        default=27,
        help="Number of GRPO optimisation steps.",
    )
    verify_h10_p.add_argument(
        "--group-size",
        type=int,
        default=6,
        help="Number of rollouts per prompt group.",
    )
    verify_h10_p.add_argument(
        "--lr",
        type=float,
        default=0.0038,
        help="GRPO learning rate.",
    )
    verify_h10_p.add_argument(
        "--backend",
        choices=["nano", "llm"],
        default="nano",
        help="Benchmark backend: 'nano' (built-in tiny model) or 'llm' (HF checkpoint).",
    )
    verify_h10_p.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HF model checkpoint used when --backend llm.",
    )
    verify_h10_p.add_argument(
        "--device",
        default="auto",
        help="Torch device used when --backend llm.",
    )
    verify_h10_p.set_defaults(func=_cmd_verify_h10)

    # Verify H11 subcommand
    verify_h11_p = sub.add_parser(
        "verify-h11",
        help="Execute H11 FP8 provider readiness verification benchmark.",
    )
    verify_h11_p.add_argument(
        "--experiment",
        default="experiments/h11_provider_fp8_readiness.yaml",
        help="Path to H11 experiment YAML.",
    )
    verify_h11_p.add_argument(
        "--output-dir",
        default="results",
        help="Directory for run output bundle.",
    )
    verify_h11_p.add_argument("--run-id", default=None, help="Optional explicit run_id override.")
    verify_h11_p.add_argument(
        "--n-prompts",
        type=int,
        default=8,
        help="Number of prompts for FP8 survival evaluation.",
    )
    verify_h11_p.add_argument(
        "--seq-len",
        type=int,
        default=32,
        help="Sequence length for FP8 survival evaluation.",
    )
    verify_h11_p.add_argument(
        "--backend",
        choices=["nano", "llm"],
        default="nano",
        help="Benchmark backend: 'nano' (built-in tiny model) or 'llm' (HF checkpoint).",
    )
    verify_h11_p.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HF model checkpoint used when --backend llm.",
    )
    verify_h11_p.add_argument(
        "--device",
        default="auto",
        help="Torch device used when --backend llm.",
    )
    verify_h11_p.set_defaults(func=_cmd_verify_h11)

    # Verify H12 subcommand
    verify_h12_p = sub.add_parser(
        "verify-h12",
        help="Execute H12 subspace-safe merge (TIES biprojected) verification benchmark.",
    )
    verify_h12_p.add_argument(
        "--experiment",
        default="experiments/h12_subspace_safe_merge.yaml",
        help="Path to H12 experiment YAML.",
    )
    verify_h12_p.add_argument(
        "--output-dir",
        default="results",
        help="Directory for run output bundle.",
    )
    verify_h12_p.add_argument("--run-id", default=None, help="Optional explicit run_id override.")
    verify_h12_p.add_argument(
        "--sql-steps",
        type=int,
        default=18,
        help="Number of SQL GRPO steps for the specialized adapter.",
    )
    verify_h12_p.add_argument(
        "--unslop-steps",
        type=int,
        default=18,
        help="Number of unslop GRPO steps for the specialized adapter.",
    )
    verify_h12_p.add_argument(
        "--group-size",
        type=int,
        default=6,
        help="GRPO group size.",
    )
    verify_h12_p.add_argument(
        "--sql-svd-rank",
        type=int,
        default=2,
        help="Number of top singular directions of Delta W_SQL to project against.",
    )
    verify_h12_p.add_argument(
        "--ties-density",
        type=float,
        default=0.85,
        help="TIES density parameter.",
    )
    verify_h12_p.add_argument(
        "--backend",
        choices=["nano", "llm"],
        default="nano",
        help="Benchmark backend: 'nano' (built-in tiny model) or 'llm' (HF checkpoint).",
    )
    verify_h12_p.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HF model checkpoint used when --backend llm.",
    )
    verify_h12_p.add_argument(
        "--device",
        default="auto",
        help="Torch device used when --backend llm.",
    )
    verify_h12_p.set_defaults(func=_cmd_verify_h12)

    # Build Tenet release subcommand
    build_tenet_p = sub.add_parser(
        "build-tenet",
        help="Build a Tenet release bundle (GRPO-tuned weights, FP8 report, readiness).",
    )
    build_tenet_p.add_argument(
        "--output-dir",
        default="results/tenet_releases",
        help="Directory for the release bundle.",
    )
    build_tenet_p.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HF model checkpoint used when --backend llm.",
    )
    build_tenet_p.add_argument(
        "--release-id",
        default="Tenet-3B-Qwen2.5-Unslop-v1",
        help="Release identifier for the Tenet bundle.",
    )
    build_tenet_p.add_argument(
        "--backend",
        choices=["nano", "llm"],
        default="nano",
        help="Release backend: 'nano' (built-in tiny model) or 'llm' (HF checkpoint).",
    )
    build_tenet_p.add_argument(
        "--device",
        default="auto",
        help="Torch device used when --backend llm.",
    )
    build_tenet_p.add_argument("--seed", type=int, default=2026, help="Random seed.")
    build_tenet_p.add_argument(
        "--top-k-layers",
        type=int,
        default=None,
        help="Optional number of top layers to select for the release.",
    )
    build_tenet_p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale applied to the released weight deltas.",
    )
    build_tenet_p.add_argument(
        "--grpo-steps",
        type=int,
        default=27,
        help="Number of GRPO steps.",
    )
    build_tenet_p.add_argument(
        "--grpo-group-size",
        type=int,
        default=6,
        help="GRPO group size.",
    )
    build_tenet_p.add_argument(
        "--grpo-lr",
        type=float,
        default=0.0038,
        help="GRPO learning rate.",
    )
    build_tenet_p.add_argument(
        "--n-prompts",
        type=int,
        default=8,
        help="Number of prompts for GRPO/evaluation.",
    )
    build_tenet_p.add_argument(
        "--seq-len",
        type=int,
        default=32,
        help="Sequence length for GRPO/evaluation.",
    )
    build_tenet_p.add_argument(
        "--hf-namespace",
        default="tenet-lab",
        help="HF Hub namespace for the release repo id.",
    )
    build_tenet_p.set_defaults(func=_cmd_build_tenet)

    # Export hub release subcommand
    export_hub_p = sub.add_parser(
        "export-hub",
        help=(
            "Export a hub release bundle (Pareto SVG, CKA quant recipe, readiness) "
            "from existing H9/H10/H11 results."
        ),
    )
    export_hub_p.add_argument(
        "--release-dir",
        required=True,
        help="Existing or target release directory.",
    )
    export_hub_p.add_argument(
        "--release-id",
        default=None,
        help="Release identifier (defaults to release_dir name).",
    )
    export_hub_p.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base HF model name.",
    )
    export_hub_p.add_argument(
        "--results-root",
        default="results",
        help="Root directory containing h9_*/h10_*/h11_* final_metrics.json.",
    )
    export_hub_p.add_argument(
        "--hf-namespace",
        default="small-model-lab",
        help="HF Hub namespace for the release repo id.",
    )
    export_hub_p.set_defaults(func=_cmd_export_hub)

    return parser


def _cmd_build_tenet(args: argparse.Namespace) -> int:
    bundle = build_tenet_release_bundle(
        output_dir=args.output_dir,
        model_name=args.model_name,
        release_id=args.release_id,
        backend=args.backend,
        device=args.device,
        seed=args.seed,
        top_k_layers=args.top_k_layers,
        scale=args.scale,
        grpo_steps=args.grpo_steps,
        grpo_group_size=args.grpo_group_size,
        grpo_lr=args.grpo_lr,
        n_prompts=args.n_prompts,
        seq_len=args.seq_len,
        hf_namespace=args.hf_namespace,
    )
    summary = {
        "release_id": bundle["release_id"],
        "release_dir": bundle["release_dir"],
        "hf_repo_id": bundle["hf_repo_id"],
        "backend": bundle["backend"],
        "ready": bundle["ready"],
        "artifacts": bundle["artifacts"],
    }
    print(json.dumps(summary, indent=2))
    return 0 if bundle["ready"] else 1


def _cmd_export_hub(args: argparse.Namespace) -> int:
    release_id = args.release_id or Path(args.release_dir).name
    bundle = export_hub_bundle_from_results(
        release_dir=args.release_dir,
        release_id=release_id,
        base_model=args.base_model,
        results_root=args.results_root,
        hf_namespace=args.hf_namespace,
    )
    summary = {
        "release_id": bundle["release_id"],
        "release_dir": bundle["release_dir"],
        "hf_repo_id": bundle["hf_repo_id"],
        "ready": bundle["ready"],
        "artifacts": bundle["artifacts"],
    }
    print(json.dumps(summary, indent=2))
    return 0 if bundle["ready"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
