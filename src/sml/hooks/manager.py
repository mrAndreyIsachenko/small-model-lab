"""Activation capture and intervention hooks compatible with SML and HF Llama/Qwen models."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn

PatchFn = Callable[[torch.Tensor], torch.Tensor]


class ActivationHookManager:
    """Context-managed forward hook utility for capturing and patching module outputs.

    Works identically on `sml.model.Transformer` and Hugging Face `LlamaForCausalLM` /
    `Qwen2ForCausalLM` models because both use `layers.{i}.self_attn`, `layers.{i}.mlp`,
    and `layers.{i}` module paths.
    """

    def __init__(self, model: nn.Module, detach: bool = True, move_to_cpu: bool = False) -> None:
        self.model = model
        self.detach = detach
        self.move_to_cpu = move_to_cpu
        self.activations: dict[str, torch.Tensor] = {}
        self._modules_by_name = dict(model.named_modules())

    def resolve_module(self, name: str) -> nn.Module:
        if name in self._modules_by_name:
            return self._modules_by_name[name]
        prefixed = f"model.{name}"
        if prefixed in self._modules_by_name:
            return self._modules_by_name[prefixed]
        raise KeyError(f"Module '{name}' not found in model.")

    @contextmanager
    def capture(self, module_names: Iterable[str]) -> Any:
        """Capture forward outputs of specified modules into `self.activations`."""
        self.activations.clear()
        handles = []

        def _make_hook(key: str) -> Callable[[nn.Module, Any, Any], None]:
            def _hook(_mod: nn.Module, _inp: Any, output: Any) -> None:
                tensor = output[0] if isinstance(output, tuple) else output
                if not isinstance(tensor, torch.Tensor):
                    return
                if self.detach:
                    tensor = tensor.detach()
                if self.move_to_cpu:
                    tensor = tensor.cpu()
                self.activations[key] = tensor

            return _hook

        try:
            for name in module_names:
                mod = self.resolve_module(name)
                handles.append(mod.register_forward_hook(_make_hook(name)))
            yield self.activations
        finally:
            for handle in handles:
                handle.remove()

    @contextmanager
    def patch(self, interventions: dict[str, PatchFn | torch.Tensor]) -> Any:
        """Replace or transform outputs of specified modules during forward execution."""
        handles = []

        def _make_patch_hook(
            patch_spec: PatchFn | torch.Tensor,
        ) -> Callable[[nn.Module, Any, Any], Any]:
            def _hook(_mod: nn.Module, _inp: Any, output: Any) -> Any:
                is_tuple = isinstance(output, tuple)
                tensor = output[0] if is_tuple else output
                if isinstance(patch_spec, torch.Tensor):
                    patched = patch_spec.to(device=tensor.device, dtype=tensor.dtype)
                else:
                    patched = patch_spec(tensor)
                if is_tuple:
                    return (patched, *output[1:])
                return patched

            return _hook

        try:
            for name, spec in interventions.items():
                mod = self.resolve_module(name)
                handles.append(mod.register_forward_hook(_make_patch_hook(spec)))
            yield
        finally:
            for handle in handles:
                handle.remove()
