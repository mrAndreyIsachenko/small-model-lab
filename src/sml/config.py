"""Typed configuration dataclasses and closed-form Transformer math utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

NormType = Literal["rmsnorm", "layernorm"]
PosEncodingType = Literal["rope", "learned"]
MLPType = Literal["swiglu", "gelu"]
ScheduleType = Literal["wsd", "cosine"]
PrecisionType = Literal["fp32", "bf16", "fp16"]


@dataclass(frozen=True)
class ModelConfig:
    """Architecture specification for SML and reference GPT-2 decoder models."""

    name: str
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    max_seq_len: int = 2048
    norm_type: NormType = "rmsnorm"
    pos_encoding: PosEncodingType = "rope"
    mlp_type: MLPType = "swiglu"
    bias: bool = False
    tie_embeddings: bool = True
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def embedding_params(self) -> int:
        """Token embedding table parameters (plus untied LM head if untied)."""
        token_emb = self.vocab_size * self.d_model
        if self.tie_embeddings:
            return token_emb
        return 2 * token_emb

    @property
    def positional_embedding_params(self) -> int:
        """Learned positional embedding parameters (0 for RoPE)."""
        if self.pos_encoding == "learned":
            return self.max_seq_len * self.d_model
        return 0

    @property
    def attention_params_per_layer(self) -> int:
        """Q, K, V, and O projection parameters per Transformer block."""
        q_weights = self.d_model * self.d_model
        kv_weights = 2 * (self.d_model * self.kv_dim)
        o_weights = self.d_model * self.d_model
        weights = q_weights + kv_weights + o_weights
        if not self.bias:
            return weights
        biases = self.d_model + 2 * self.kv_dim + self.d_model
        return weights + biases

    @property
    def mlp_params_per_layer(self) -> int:
        """Feed-forward parameters per Transformer block (3 matrices for SwiGLU, 2 for GELU)."""
        if self.mlp_type == "swiglu":
            weights = 3 * (self.d_model * self.d_ff)
            biases = (2 * self.d_ff + self.d_model) if self.bias else 0
            return weights + biases
        weights = 2 * (self.d_model * self.d_ff)
        biases = (self.d_ff + self.d_model) if self.bias else 0
        return weights + biases

    @property
    def norm_params_per_layer(self) -> int:
        """Pre-attention and pre-MLP normalization parameters per layer."""
        params_per_norm = self.d_model * (2 if self.norm_type == "layernorm" else 1)
        return 2 * params_per_norm

    @property
    def final_norm_params(self) -> int:
        return self.d_model * (2 if self.norm_type == "layernorm" else 1)

    @property
    def non_embedding_params(self) -> int:
        """All Transformer block, final norm, and learned position parameters."""
        per_layer = (
            self.attention_params_per_layer + self.mlp_params_per_layer + self.norm_params_per_layer
        )
        return self.n_layers * per_layer + self.final_norm_params + self.positional_embedding_params

    @property
    def total_params(self) -> int:
        """Exact total trainable parameter count."""
        return self.embedding_params + self.non_embedding_params

    def kv_cache_bytes_per_token(self, bytes_per_element: int = 2) -> int:
        """KV-cache memory footprint per token across all layers (default bf16/fp16 = 2 bytes)."""
        return 2 * self.n_layers * self.kv_dim * bytes_per_element

    def flops_per_token(self, seq_len: int | None = None) -> int:
        """Estimated forward + backward training FLOPs per token.

        Uses 6 * N_total (capturing linear weights and tied LM head matmul)
        plus 6 * n_layers * d_model * T for causal attention score/value matmuls.
        """
        t = seq_len if seq_len is not None else self.max_seq_len
        linear_flops = 6 * self.total_params
        causal_attn_flops = 6 * self.n_layers * self.d_model * t
        return linear_flops + causal_attn_flops


@dataclass(frozen=True)
class HardwareProfile:
    """Hardware pricing and theoretical compute specification."""

    name: str
    accelerator: Literal["cuda", "mps", "cpu"]
    vram_gb: float
    peak_bf16_tflops: float
    hourly_rate_usd: float
    price_as_of: str
    provider_notes: str = ""


@dataclass(frozen=True)
class DataConfig:
    """Dataset source and sharding configuration."""

    name: str
    hf_dataset: str
    hf_subset: str
    vocab_size: int
    train_tokens: int
    val_tokens: int
    test_tokens: int
    shard_size_tokens: int
    hf_repo_id: str
    license: str = "ODC-By-1.0"


@dataclass(frozen=True)
class TrainConfig:
    """Optimizer, schedule, and batching hyperparameters."""

    batch_size_tokens: int
    micro_batch_size: int
    seq_len: int
    max_steps: int
    lr: float
    min_lr_ratio: float = 0.0
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    schedule: ScheduleType = "wsd"
    warmup_steps: int = 200
    decay_fraction: float = 0.15
    branch_cooldown_tokens: int | None = None
    precision: PrecisionType = "bf16"
    compile: bool = True

    @property
    def tokens_per_micro_step(self) -> int:
        return self.micro_batch_size * self.seq_len

    @property
    def grad_accum_steps(self) -> int:
        if self.batch_size_tokens % self.tokens_per_micro_step != 0:
            raise ValueError(
                f"batch_size_tokens ({self.batch_size_tokens}) must be divisible by "
                f"micro_batch_size * seq_len ({self.tokens_per_micro_step})"
            )
        return self.batch_size_tokens // self.tokens_per_micro_step

    @property
    def total_planned_tokens(self) -> int:
        return self.batch_size_tokens * self.max_steps


@dataclass(frozen=True)
class ExperimentSpec:
    """Complete pre-registered experiment specification."""

    experiment_id: str
    hypothesis_id: str
    hypothesis: str
    prediction: str
    acceptance_criterion: str
    seed: int
    model: ModelConfig
    data: DataConfig
    hardware: HardwareProfile
    train: TrainConfig
    model_config_path: str
    data_config_path: str
    hardware_config_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(data).__name__}")
    return data


def load_model_config(path: str | Path) -> ModelConfig:
    return ModelConfig(**load_yaml(path))


def load_hardware_profile(path: str | Path) -> HardwareProfile:
    return HardwareProfile(**load_yaml(path))


def load_data_config(path: str | Path) -> DataConfig:
    return DataConfig(**load_yaml(path))


def load_experiment_spec(path: str | Path, repo_root: Path | None = None) -> ExperimentSpec:
    exp_path = Path(path)
    root = repo_root if repo_root is not None else exp_path.resolve().parent.parent
    raw = load_yaml(exp_path)

    model_cfg = load_model_config(root / raw["model_config_path"])
    data_cfg = load_data_config(root / raw["data_config_path"])
    hw_cfg = load_hardware_profile(root / raw["hardware_config_path"])
    train_cfg = TrainConfig(**raw["train"])

    return ExperimentSpec(
        experiment_id=raw["experiment_id"],
        hypothesis_id=raw["hypothesis_id"],
        hypothesis=str(raw["hypothesis"]).strip(),
        prediction=str(raw["prediction"]).strip(),
        acceptance_criterion=str(raw["acceptance_criterion"]).strip(),
        seed=int(raw["seed"]),
        model=model_cfg,
        data=data_cfg,
        hardware=hw_cfg,
        train=train_cfg,
        model_config_path=raw["model_config_path"],
        data_config_path=raw["data_config_path"],
        hardware_config_path=raw["hardware_config_path"],
    )
