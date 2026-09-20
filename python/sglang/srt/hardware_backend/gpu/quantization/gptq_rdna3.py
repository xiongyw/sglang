"""Dense GPTQ runtime adapter for the RX 7900 XTX appliance."""

from __future__ import annotations

from typing import Optional

import torch


def _is_identity_group_index(g_idx: torch.Tensor, group_size: int) -> bool:
    """Whether ``g_idx`` is the identity grouping, allowing a TP shard offset.

    A checkpoint with no activation reordering stores ``g_idx[i] = i // group_size``
    over the full K dimension. Under tensor parallelism a row-parallel layer loads
    only its own K-shard, so the surviving values are the same identity shifted by
    the shard's first group; that is still identity ordering. A genuine act-order
    permutation is not a contiguous blocked ramp and is rejected.
    """
    if g_idx.numel() == 0:
        return True
    expected = torch.arange(
        g_idx.numel(), device=g_idx.device, dtype=torch.int32
    ) // group_size
    return bool(torch.equal(g_idx, expected + g_idx[0]))


class GPTQLinearKernel:
    """Run the target checkpoint's identity-g_idx W4A16 tensors on gfx1100."""

    def __init__(self, quant_config) -> None:
        if quant_config.weight_bits != 4 or quant_config.group_size != 64:
            raise ValueError(
                "7900xtx-qwen38-27b requires GPTQ 4-bit weights with group_size=64"
            )
        if quant_config.desc_act:
            raise ValueError(
                "7900xtx-qwen38-27b does not support act-order GPTQ"
            )
        self.quant_config = quant_config
        self.use_shuffle = True
        self.use_v2_format = quant_config.checkpoint_format == "gptq_v2"

    @staticmethod
    def _operators():
        from sgl_kernel import common_ops  # noqa: F401

        return torch.ops.sgl_kernel

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        for name in ("qweight", "qzeros", "g_idx", "scales"):
            parameter = getattr(layer, name)
            data = parameter.data
            if name == "scales" and data.dtype == torch.bfloat16:
                data = data.to(torch.float16)
            setattr(layer, name, torch.nn.Parameter(data, requires_grad=False))
        if not _is_identity_group_index(layer.g_idx, self.quant_config.group_size):
            raise ValueError(
                "7900xtx-qwen38-27b requires identity GPTQ g_idx for the target "
                f"checkpoint (got {layer.g_idx.numel()} entries spanning groups "
                f"{int(layer.g_idx.min())}..{int(layer.g_idx.max())} with "
                f"group_size={self.quant_config.group_size})"
            )
        layer.g_idx = torch.nn.Parameter(
            torch.empty((0,), dtype=torch.int32, device=layer.qweight.device),
            requires_grad=False,
        )
        self._operators().gptq_shuffle_rdna3(layer.qweight, layer.g_idx)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output_shape = x.shape[:-1] + (layer.qweight.shape[-1],)
        activations = x.reshape(-1, x.shape[-1]).contiguous()
        output = self._operators().gptq_gemm_rdna3(
            activations,
            layer.qweight,
            layer.qzeros,
            layer.scales,
            layer.g_idx,
            self.use_v2_format,
        )
        if bias is not None:
            output.add_(bias)
        return output.reshape(output_shape)
