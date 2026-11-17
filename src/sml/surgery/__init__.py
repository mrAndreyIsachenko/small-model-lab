"""Surgery module for weight modification, CKA drift, quantization, and Pareto analysis."""

from .cka import (
    collect_layer_activations,
    compute_layerwise_cka_drift,
    get_transformer_layers,
    linear_cka,
)
from .pareto import (
    compute_logit_kl_divergence,
    run_h9_biprojected_surgery_benchmark,
    run_llm_h9_biprojected_surgery_benchmark,
    run_llm_surgery_and_pareto_suite,
    run_surgery_and_pareto_suite,
)
from .probes import (
    REASONING_ANCHOR_PAIRS,
    REFUSAL_CONTRAST_PAIRS,
    SLOP_CONTRAST_PAIRS,
    encode_contrast_pairs,
)
from .quantization import (
    allocate_mixed_precision_bits,
    apply_mixed_precision_quantization,
    prune_transformer_layers,
    quantize_tensor_groupwise,
)
from .subspace import (
    LancetRecipe,
    apply_lancet_biprojected_surgery,
    apply_lancet_recipe,
    apply_subspace_ablation,
    biproject_direction,
    build_lancet_recipe,
    compute_layer_scr_profile,
    extract_contrastive_direction,
    load_lancet_recipe,
    project_out_rank1_subspace,
    project_out_subspace_norm_preserving,
    save_lancet_recipe,
)

__all__ = [
    "REFUSAL_CONTRAST_PAIRS",
    "REASONING_ANCHOR_PAIRS",
    "SLOP_CONTRAST_PAIRS",
    "LancetRecipe",
    "allocate_mixed_precision_bits",
    "apply_lancet_biprojected_surgery",
    "apply_lancet_recipe",
    "apply_mixed_precision_quantization",
    "apply_subspace_ablation",
    "biproject_direction",
    "build_lancet_recipe",
    "collect_layer_activations",
    "compute_layer_scr_profile",
    "compute_layerwise_cka_drift",
    "compute_logit_kl_divergence",
    "encode_contrast_pairs",
    "extract_contrastive_direction",
    "get_transformer_layers",
    "linear_cka",
    "load_lancet_recipe",
    "project_out_rank1_subspace",
    "project_out_subspace_norm_preserving",
    "prune_transformer_layers",
    "quantize_tensor_groupwise",
    "run_h9_biprojected_surgery_benchmark",
    "run_llm_h9_biprojected_surgery_benchmark",
    "run_llm_surgery_and_pareto_suite",
    "run_surgery_and_pareto_suite",
    "save_lancet_recipe",
]
