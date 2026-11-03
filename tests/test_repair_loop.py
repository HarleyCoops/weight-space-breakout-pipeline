import json
import unittest
from weight_space_breakout.policy import faulty_policy, repair_edits
from weight_space_breakout.repair_loop import fingerprint, from_sparse, passes_gate, verify_round


class LoopTests(unittest.TestCase):
    def test_snapshot_roundtrip_and_bounds(self):
        policy = faulty_policy("reversed_controls")
        self.assertEqual(fingerprint(policy), fingerprint(from_sparse(json.dumps(policy.sparse()))))
        with self.assertRaises(ValueError):
            from_sparse('[{"tensor":"W1","index":[500,0],"value":1}]')

    def test_gate_requires_mean_uncertainty_and_actual_change(self):
        self.assertTrue(passes_gate({"gain":1.0,"paired_standard_error":0.1}, 2))
        self.assertFalse(passes_gate({"gain":1.0,"paired_standard_error":0.1}, 0))
        self.assertFalse(passes_gate({"gain":0.05,"paired_standard_error":0.0}, 2))
        self.assertFalse(passes_gate({"gain":0.2,"paired_standard_error":0.2}, 2))
        self.assertFalse(passes_gate({"gain":-1.0,"paired_standard_error":0.1}, 2))

    def test_invalid_edits_keep_incumbent(self):
        policy = faulty_policy("reversed_controls")
        result = verify_round('not JSON', policy, 0)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["selected_hash"], fingerprint(policy))

    def test_known_repair_passes_heldout_gate(self):
        policy = faulty_policy("reversed_controls")
        result = verify_round(json.dumps(repair_edits("reversed_controls")), policy, 0)
        self.assertTrue(result["accepted"])
        self.assertNotEqual(result["selected_hash"], fingerprint(policy))
        self.assertEqual(result["gate"]["episodes"], 512)
        self.assertEqual(result["gate"]["seed_start"], 60000)
        self.assertNotIn("final_audit", result)


if __name__ == "__main__":
    unittest.main()
