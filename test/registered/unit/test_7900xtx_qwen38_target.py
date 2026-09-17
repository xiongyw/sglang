from __future__ import annotations

import json
import unittest
from pathlib import Path

from tools.target_profiles.qwen38_7900xtx import (
    TARGET_PROFILE_NAME,
    TargetContractError,
    evaluate_target_contract,
    load_checkpoint_signature,
)


TARGET_GPU = {
    "gcn_arch": "gfx1100",
    "compute_units": 96,
    "total_memory_bytes": 25_753_026_560,
    "hip_version": "7.2.26015",
}

TARGET_CHECKPOINT = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "text_config": {
        "model_type": "qwen3_5_text",
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_hidden_layers": 64,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
    },
    "quantization_config": {
        "quant_method": "gptq",
        "bits": 4,
        "group_size": 64,
        "desc_act": False,
        "sym": True,
    },
}


class TestQwen38TargetContract(unittest.TestCase):
    def test_exact_target_contract_matches(self) -> None:
        result = evaluate_target_contract(TARGET_GPU, TARGET_CHECKPOINT)

        self.assertTrue(result.matches)
        self.assertEqual(result.profile_name, TARGET_PROFILE_NAME)
        self.assertEqual(result.mismatches, ())

    def test_contract_rejects_same_isa_with_wrong_compute_units(self) -> None:
        gpu = {**TARGET_GPU, "compute_units": 60}

        result = evaluate_target_contract(gpu, TARGET_CHECKPOINT)

        self.assertFalse(result.matches)
        self.assertIn("compute_units: expected 96, got 60", result.mismatches)

    def test_contract_rejects_wrong_gptq_layout(self) -> None:
        checkpoint = {
            **TARGET_CHECKPOINT,
            "quantization_config": {
                **TARGET_CHECKPOINT["quantization_config"],
                "group_size": 128,
            },
        }

        result = evaluate_target_contract(TARGET_GPU, checkpoint)

        self.assertFalse(result.matches)
        self.assertIn(
            "quantization_config.group_size: expected 64, got 128", result.mismatches
        )

    def test_required_contract_raises_actionable_error(self) -> None:
        gpu = {**TARGET_GPU, "gcn_arch": "gfx1151"}

        with self.assertRaisesRegex(
            TargetContractError,
            "gcn_arch: expected 'gfx1100', got 'gfx1151'",
        ):
            evaluate_target_contract(gpu, TARGET_CHECKPOINT, require_match=True)

    def test_load_checkpoint_signature_reads_target_config(self) -> None:
        with self.subTest("valid config"):
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as directory:
                config_path = Path(directory) / "config.json"
                config_path.write_text(json.dumps(TARGET_CHECKPOINT), encoding="utf-8")

                signature = load_checkpoint_signature(Path(directory))

        self.assertEqual(signature["architectures"], ["Qwen3_5ForConditionalGeneration"])
        self.assertEqual(signature["quantization_config"]["group_size"], 64)


if __name__ == "__main__":
    unittest.main()
