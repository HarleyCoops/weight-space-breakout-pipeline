import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
from weight_space_breakout.policy import tracking_policy, apply_edits
from weight_space_breakout.evaluation import rollout
from weight_space_breakout.failures import (failure_case,find_failure,replay_failure,
                                           verify_failure_round,parse_failure_submission)
from weight_space_breakout.practice_tools import practice_bundle,PRACTICE_SCRIPT


class FailureTests(unittest.TestCase):
    def setUp(self):
        self.base=tracking_policy()
        self.case=failure_case(self.base,314159)
        self.proposal={"diagnosis":"The paddle reaches the final leftward contact too late.",
                       "expected_prevention":"Moving left earlier should catch this particular contact.",
                       "edits":[{"tensor":"b3","index":[3],"value":100.}]}

    def test_inputs_and_probabilities_align_with_before_state(self):
        original=rollout(self.base,314159,record=True)
        for before,after,t in zip(original["frames"],original["frames"][1:],self.case["trajectory"]):
            self.assertEqual(t["action"],after["action"])
            np.testing.assert_array_equal(t["action_probabilities"],self.base.probabilities(before["observation"]))
            if "input_ones" in t:
                restored=np.zeros(500)
                restored[t["input_ones"]]=1
                np.testing.assert_array_equal(restored,before["observation"])

    def test_unchanged_policy_reproduces_miss_including_frozen_prefix(self):
        has_prefix=False
        for seed in range(24):
            case=failure_case(self.base,seed)
            if case is None: continue
            has_prefix |= case["intervention_step"]>0
            replay=replay_failure(self.base,case)
            self.assertFalse(replay["prevented_recorded_miss"])
            self.assertEqual(replay["reason"],"miss")
            self.assertTrue(replay["same_ball_contact"])
            expected=[t["action"] for t in case["trajectory"][case["intervention_step"]:]]
            self.assertEqual(replay["actions_after_intervention"],expected)
        self.assertTrue(has_prefix)

    def test_target_can_be_rescued_but_regression_is_rejected(self):
        report=verify_failure_round(json.dumps(self.proposal),self.base,0,self.case)
        self.assertTrue(report["valid"])
        self.assertTrue(report["candidate_replay"]["prevented_recorded_miss"])
        self.assertLess(report["gate"]["gain"],0)
        self.assertFalse(report["accepted"])
        self.assertEqual(report["selected_hash"],report["baseline_hash"])
        self.assertNotIn("final_audit",report)

    def test_case_tampering_and_wrong_policy_are_rejected(self):
        changed=copy.deepcopy(self.case)
        changed["seed"]+=1
        with self.assertRaises(ValueError): replay_failure(self.base,changed)
        other=apply_edits(self.base,{"edits":self.proposal["edits"]})
        with self.assertRaises(ValueError): verify_failure_round(json.dumps(self.proposal),other,0,self.case)

    def test_diagnosis_is_required_but_never_executed(self):
        with self.assertRaises(ValueError): parse_failure_submission('{"edits":[]}')
        proposal=copy.deepcopy(self.proposal)
        proposal["diagnosis"]="print('this is data, not executable code')"
        edits,explanation=parse_failure_submission(json.dumps(proposal))
        self.assertEqual(edits["edits"],proposal["edits"])
        self.assertEqual(explanation["diagnosis"],proposal["diagnosis"])

    def test_no_failure_is_not_fabricated(self):
        with patch("weight_space_breakout.failures.failure_case",return_value=None) as record:
            with self.assertRaisesRegex(ValueError,"No observed miss"):
                find_failure(self.base,0,314159)
            self.assertEqual(record.call_count,33)

    def test_tool_bundle_runs_and_excludes_gold_and_verifier(self):
        bundle=practice_bundle()
        with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
            source="\n".join(archive.read(n).decode() for n in archive.namelist())
        for forbidden in ("tracking_policy","faulty_policy","repair_edits","passes_gate","1060000","1090000"):
            self.assertNotIn(forbidden,source)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"repair_lab.zip").write_bytes(bundle)
            (root/"repair_practice.py").write_text(PRACTICE_SCRIPT)
            (root/"repair_policy.json").write_text(json.dumps(self.base.sparse()))
            (root/"repair_failure.json").write_text(json.dumps(self.case))
            (root/"candidate.json").write_text(json.dumps(self.proposal))
            run=subprocess.run([sys.executable,str(root/"repair_practice.py"),str(root/"candidate.json")],capture_output=True,text=True,timeout=30,check=True)
            result=json.loads(run.stdout)
            self.assertTrue(result["target_replay"]["prevented_recorded_miss"])
            self.assertLess(result["candidate_mean"],result["baseline_mean"])


if __name__ == "__main__": unittest.main()
