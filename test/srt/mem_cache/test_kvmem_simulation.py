import unittest

from sglang.srt.mem_cache.kvmem_selector import (
    KVMemBlock,
    KVMemBlockSelector,
    KVMemSelectionConfig,
)
from sglang.srt.mem_cache.kvmem_simulation import KVMemResidentSimulation


class TestKVMemResidentSimulation(unittest.TestCase):
    def test_tracks_reuse_and_transfer_churn_between_rounds(self):
        blocks = [KVMemBlock(i, i * 10, 10) for i in range(6)]
        selector = KVMemBlockSelector(
            KVMemSelectionConfig(budget_tokens=30, sink_blocks=1)
        )
        sim = KVMemResidentSimulation(selector)

        first = sim.step(blocks, {1: 3.0, 2: 2.0, 3: 1.0})
        second = sim.step(blocks, {1: 1.0, 2: 2.0, 4: 9.0})

        self.assertEqual(first.selected, [0, 1, 2])
        self.assertEqual(first.stage_in, [0, 1, 2])
        self.assertEqual(first.reused, 0)
        self.assertEqual(second.selected, [0, 2, 4])
        self.assertEqual(second.stage_in, [4])
        self.assertEqual(second.stage_out, [1])
        self.assertEqual(second.reused, 2)
        self.assertEqual(sim.total_steps, 2)
        self.assertEqual(sim.total_stage_in, 4)
        self.assertEqual(sim.total_stage_out, 1)

    def test_reset_clears_residency_and_counters(self):
        blocks = [KVMemBlock(i, i, 1) for i in range(3)]
        sim = KVMemResidentSimulation(
            KVMemBlockSelector(KVMemSelectionConfig(budget_tokens=2))
        )
        sim.step(blocks, {1: 1.0})
        sim.reset()
        result = sim.step(blocks, {2: 1.0})

        self.assertEqual(result.stage_in, [0, 2])
        self.assertEqual(sim.total_steps, 1)


if __name__ == "__main__":
    unittest.main()
