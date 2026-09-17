from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "python"
    / "sglang"
    / "kernels"
    / "aot"
    / "rdna_build_plan.py"
)


def load_build_plan_module():
    spec = importlib.util.spec_from_file_location("rdna_build_plan", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Test7900XtxAotBuildPlan(unittest.TestCase):
    def test_gfx1100_uses_rdna_build_plan(self) -> None:
        module = load_build_plan_module()
        sources = [
            "csrc/allreduce/custom_all_reduce.hip",
            "csrc/allreduce/deterministic_all_reduce.hip",
            "csrc/allreduce/quick_all_reduce.cu",
            "csrc/common_extension_rocm.cc",
        ]

        plan = module.make_build_plan("gfx1100", sources)

        self.assertTrue(plan.is_target)
        self.assertEqual(plan.sources, ("csrc/common_extension_rocm.cc",))
        self.assertEqual(plan.topk_dynamic_smem_bytes, 48 * 1024)
        self.assertIn("-DSGL_IS_RDNA", plan.hipcc_flags)
        self.assertIn("-DSGL_IS_RDNA", plan.cxx_flags)

    def test_non_target_is_rejected(self) -> None:
        module = load_build_plan_module()

        with self.assertRaisesRegex(ValueError, "expects gfx1100"):
            module.make_build_plan("gfx1151", ["csrc/common_extension_rocm.cc"])

    def test_allreduce_filter_keeps_non_allreduce_sources(self) -> None:
        module = load_build_plan_module()
        sources = [
            "csrc/allreduce/custom_all_reduce.hip",
            "csrc/gemm/gptq/q_gemm_rdna3.cu",
            "csrc/elementwise/topk.hip",
        ]

        plan = module.make_build_plan("gfx1100", sources)

        self.assertEqual(
            plan.sources,
            (
                "csrc/gemm/gptq/q_gemm_rdna3.cu",
                "csrc/elementwise/topk.hip",
            ),
        )


if __name__ == "__main__":
    unittest.main()
