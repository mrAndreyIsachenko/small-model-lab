"""Activation capture and patching hooks compatible with SML and Hugging Face Llama/Qwen models."""

from sml.hooks.manager import ActivationHookManager, PatchFn

__all__ = ["ActivationHookManager", "PatchFn"]
