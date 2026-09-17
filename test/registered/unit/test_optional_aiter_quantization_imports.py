from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class TestOptionalAiterQuantizationImports(unittest.TestCase):
    def test_quantization_registry_imports_without_aiter(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from sglang.srt.layers.quantization import get_quantization_config; "
                "print(get_quantization_config('auto-round').__name__)",
            ],
            cwd=REPOSITORY_ROOT,
            env={"PYTHONPATH": str(REPOSITORY_ROOT / "python")},
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("AutoRoundConfig", result.stdout)


if __name__ == "__main__":
    unittest.main()
