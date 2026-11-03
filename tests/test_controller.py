"""Controller boundary checks using a compact synthetic trace, without game execution."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from pipeline import control
from weight_space_breakout.policy import faulty_policy, apply_edits, repair_edits
from weight_space_breakout.repair_loop import fingerprint


def fixture():
    baseline = faulty_policy("reversed_controls")
    edits = repair_edits("reversed_controls")
    candidate = apply_edits(baseline,edits)
    plan = {"environment":"owner/weight-space-breakout@0.2.1", "editor":"test-editor", "rounds":3,
            "temperature":0,"max_tokens":4096}
    submission = {"evaluation_id":"eval-1","baseline_hash":fingerprint(baseline)}
    report = {"round_index":0, "baseline_hash":fingerprint(baseline),"candidate_hash":fingerprint(candidate),
              "selected_hash":fingerprint(candidate), "valid":True,"accepted":True,
              "changed_parameters":2,"edits":edits,"verifier":"weight-space-breakout@0.2.1",
              "gate":{"episodes":512,"seed_start":60000,"seed_end":60511,
                      "baseline_returns":[1]*512,"candidate_returns":[2]*512,
                      "baseline_return":1.,"candidate_return":2.,"gain":1.,"paired_standard_error":0.}}
    summary = {"environment_id":plan["environment"],"activity":{"model_calls":1,"tool_calls":0,"tool_results":0},
               "execution":{"has_error":False,"is_truncated":False}}
    trace = {"id":"trace-1","run":{"id":"eval-1"},"ok":True,"is_completed":True,"errors":[],
             "calls":[{"model":plan["editor"],"finish_reason":"stop","sampling":{"temperature":0,"max_tokens":4096}}],
             "nodes":[{"message":{"role":"assistant","content":json.dumps(edits)}}],
             "info":{"repair_pipeline":report},"rewards":{"improvement":{"score":1.}},
             "task":{"data":{"round_index":0,"loop_policy":json.dumps(baseline.sparse()),"final_round":False,"prompt":"Fixture"}}}
    return trace, summary, submission, plan, baseline, 0


class ControllerTests(unittest.TestCase):
    def test_valid_result_and_tampered_report(self):
        args = fixture()
        self.assertEqual(fingerprint(control.validate_trace(*args)[1]),args[0]["info"]["repair_pipeline"]["selected_hash"])
        for key,value in (("candidate_hash","wrong"),("accepted",False),("changed_parameters",3)):
            altered = copy.deepcopy(args)
            altered[0]["info"]["repair_pipeline"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                control.validate_trace(*altered)

    def test_tool_use_and_wrong_model_are_not_accepted(self):
        args = fixture()
        args[1]["activity"]["tool_calls"] = 1
        with self.assertRaises(ValueError): control.validate_trace(*args)
        args = fixture()
        args[0]["calls"][0]["model"] = "another-editor"
        with self.assertRaises(ValueError): control.validate_trace(*args)

    def test_metrics_must_match_per_episode_returns(self):
        args = fixture()
        args[0]["info"]["repair_pipeline"]["gate"]["candidate_returns"][0] = 20
        with self.assertRaises(ValueError): control.validate_trace(*args)

    def test_early_final_audit_is_rejected(self):
        args = fixture()
        args[0]["info"]["repair_pipeline"]["final_audit"] = {}
        with self.assertRaises(ValueError): control.validate_trace(*args)

    def test_snapshot_recovers_interrupted_state_write(self):
        baseline = faulty_policy("reversed_controls")
        candidate = apply_edits(baseline,repair_edits("reversed_controls"))
        state = {"current_hash":fingerprint(candidate),"history":[{"round":0,"accepted":True,
                 "baseline_hash":fingerprint(baseline),"selected_hash":fingerprint(candidate)}]}
        with tempfile.TemporaryDirectory() as folder, patch.object(control,"RUN",Path(folder)):
            control.save(Path(folder)/"current-policy.json",baseline.sparse())
            control.save(Path(folder)/"round-00/candidate-policy.json",candidate.sparse())
            self.assertEqual(fingerprint(control.current_policy(state)),fingerprint(candidate))
            self.assertEqual(fingerprint(control.current_policy(state)),fingerprint(candidate))


if __name__ == "__main__": unittest.main()
