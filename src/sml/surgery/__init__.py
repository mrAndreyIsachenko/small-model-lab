"""Surgery module for weight modification, CKA drift, quantization, and Pareto analysis."""

from .cka import (
    collect_layer_activations,
    compute_layerwise_cka_drift,
    linear_cka,
)
from .pareto import (
    compute_logit_kl_divergence,
    run_surgery_and_pareto_suite,
)
from .quantization import (
    allocate_mixed_precision_bits,
    apply_mixed_precision_quantization,
    prune_transformer_layers,
    quantize_tensor_groupwise,
)
from .subspace import (
    apply_subspace_ablation,
    extract_contrastive_direction,
    project_out_rank1_subspace,
)

__all__ = [
    "allocate_mixed_precision_bits",
    "apply_mixed_precision_quantization",
    "apply_subspace_ablation",
    "collect_layer_activations",
    "compute_layerwise_cka_drift",
    "compute_logit_kl_divergence",
    "extract_contrastive_direction",
    "linear_cka",
    "project_out_rank1_subspace",
    "prune_transformer_layers",
    "quantize_tensor_groupwise",
    "run_surgery_and_pareto_suite",
]
