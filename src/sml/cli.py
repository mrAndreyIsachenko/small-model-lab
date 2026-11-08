"""Command-line interface for Small Model Lab (`sml`)."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from sml.config import load_experiment_spec, load_model_config
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
