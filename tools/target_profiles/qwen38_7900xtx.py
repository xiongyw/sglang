"""Exact runtime/checkpoint contract for the RX 7900 XTX Qwen3.8 appliance.

This module intentionally describes one measured profile. It is not a generic
RDNA capability detector and it does not select a fallback implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

TARGET_PROFILE_NAME = "rx7900xtx-qwen38-27b-w4a16-rocm724"
_TARGET_GPU = {
    "gcn_arch": "gfx1100",
    "compute_units": 96,
    "total_memory_bytes": 25_753_026_560,
    "hip_version_prefix": "7.2.",
}
_TARGET_TEXT_CONFIG = {
    "model_type": "qwen3_5_text",
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_hidden_layers": 64,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
}
_TARGET_QUANTIZATION = {
    "quant_method": "gptq",
    "bits": 4,
    "group_size": 64,
    "desc_act": False,
    "sym": True,
}


class TargetContractError(RuntimeError):
    """Raised when the specialized appliance contract does not match."""


@dataclass(frozen=True)
class TargetContractResult:
    profile_name: str
    mismatches: tuple[str, ...]

    @property
    def matches(self) -> bool:
        return not self.mismatches


def _expect(
    mismatches: list[str], source: Mapping[str, Any], key: str, expected: Any, prefix: str = ""
) -> None:
    actual = source.get(key)
    if actual != expected:
        name = f"{prefix}{key}"
        mismatches.append(f"{name}: expected {expected!r}, got {actual!r}")


def load_checkpoint_signature(model_path: Path) -> dict[str, Any]:
    """Read the subset of config.json that identifies the target checkpoint layout."""

    config_path = model_path / "config.json"
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    return {
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "text_config": config.get("text_config") or {},
        "quantization_config": config.get("quantization_config") or {},
    }


def evaluate_target_contract(
    gpu: Mapping[str, Any], checkpoint: Mapping[str, Any], *, require_match: bool = False
) -> TargetContractResult:
    """Compare observable runtime and checkpoint properties with this appliance."""

    mismatches: list[str] = []
    _expect(mismatches, gpu, "gcn_arch", _TARGET_GPU["gcn_arch"])
    _expect(mismatches, gpu, "compute_units", _TARGET_GPU["compute_units"])
    _expect(mismatches, gpu, "total_memory_bytes", _TARGET_GPU["total_memory_bytes"])
    hip_version = gpu.get("hip_version")
    if not isinstance(hip_version, str) or not hip_version.startswith(
        _TARGET_GPU["hip_version_prefix"]
    ):
        mismatches.append(
            "hip_version: expected prefix "
            f"{_TARGET_GPU['hip_version_prefix']!r}, got {hip_version!r}"
        )

    _expect(mismatches, checkpoint, "architectures", ["Qwen3_5ForConditionalGeneration"])
    _expect(mismatches, checkpoint, "model_type", "qwen3_5")
    text_config = checkpoint.get("text_config")
    if not isinstance(text_config, Mapping):
        mismatches.append("text_config: expected mapping, got " f"{type(text_config).__name__}")
    else:
        for key, expected in _TARGET_TEXT_CONFIG.items():
            _expect(mismatches, text_config, key, expected, "text_config.")
    quantization_config = checkpoint.get("quantization_config")
    if not isinstance(quantization_config, Mapping):
        mismatches.append(
            "quantization_config: expected mapping, got "
            f"{type(quantization_config).__name__}"
        )
    else:
        for key, expected in _TARGET_QUANTIZATION.items():
            _expect(mismatches, quantization_config, key, expected, "quantization_config.")

    result = TargetContractResult(TARGET_PROFILE_NAME, tuple(mismatches))
    if require_match and not result.matches:
        raise TargetContractError(
            f"{TARGET_PROFILE_NAME} target contract mismatch: " + "; ".join(result.mismatches)
        )
    return result


def discover_local_gpu() -> dict[str, Any]:
    """Return properties needed to evaluate the active HIP device."""

    import torch

    if not torch.cuda.is_available() or not torch.version.hip:
        raise TargetContractError("target contract requires an available ROCm/HIP device")
    properties = torch.cuda.get_device_properties(0)
    return {
        "gcn_arch": properties.gcnArchName,
        "compute_units": properties.multi_processor_count * 2,
        "total_memory_bytes": properties.total_memory,
        "hip_version": torch.version.hip,
    }


def main() -> int:
    """Print the contract decision for a local model path."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    arguments = parser.parse_args()
    result = evaluate_target_contract(discover_local_gpu(), load_checkpoint_signature(arguments.model_path))
    print(json.dumps({"profile": result.profile_name, "matches": result.matches, "mismatches": result.mismatches}))
    return 0 if result.matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
