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
from weight_space_breakout.failures import failure_case, replay_failure


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
    def exhausted_fixture(self):
        args=self.failure_fixture()
        t,s,_,plan,base,_=args
        s["execution"]["is_truncated"]=True
        s["score"]={"outcome":"max_turns","reward":0.}
        s["activity"].update(model_calls=8,tool_calls=8,tool_results=7)
        t["stop_condition"]="max_turns"
        call=copy.deepcopy(t["calls"][0])
        call["finish_reason"]="tool_calls"
        t["calls"]=[copy.deepcopy(call) for _ in range(8)]
        # Tool-only messages may omit content entirely (as in the real Prime trace).
        t["nodes"]=[{"message":{"role":"assistant","tool_calls":[{"name":"bash","arguments":"{}"}]}} for _ in range(8)]
        t["nodes"].insert(1,{"message":{"role":"tool","content":"AttributeError: 'Policy' object has no attribute 'W1'\n"}})
        r=t["info"]["repair_pipeline"]
        r.update(valid=False,accepted=False,reward=0.,error="Expecting value",selected_hash=fingerprint(base))
        for key in ("edits","candidate_hash","candidate_replay","explanation","changed_parameters","gate"):
            r.pop(key,None)
        return args

    def test_expected_turn_exhaustion_records_rejection(self):
        args=self.exhausted_fixture()
        _,selected,report,editor=control.validate_trace(*args)
        self.assertFalse(report["accepted"])
        self.assertEqual(fingerprint(selected),fingerprint(args[4]))
        self.assertEqual(editor["outcome"],"turn_limit_without_submission")
        self.assertIn("AttributeError",editor["tool_errors"][0])

    def test_truncated_candidate_and_other_errors_remain_blocked(self):
        args=self.exhausted_fixture()
        args[0]["info"]["repair_pipeline"]["valid"]=True
        with self.assertRaises(ValueError): control.validate_trace(*args)
        args=self.exhausted_fixture()
        args[1]["execution"]["has_error"]=True
        with self.assertRaises(ValueError): control.validate_trace(*args)
        args=self.exhausted_fixture()
        args[1]["score"]["outcome"]="timeout"
        with self.assertRaises(ValueError): control.validate_trace(*args)
        args=self.exhausted_fixture()
        args[0]["info"]["repair_pipeline"]["selected_hash"]="another-policy"
        with self.assertRaises(ValueError): control.validate_trace(*args)

    def failure_fixture(self):
        args=fixture()
        t,s,submission,plan,base,index=args
        plan.update(protocol_id="failure-driven-v1",max_model_calls=8,max_tool_calls=24,max_output_tokens=16384)
        case=failure_case(base,314159)
        t["task"]["data"]["failure_case_json"]=json.dumps(case)
        explanation={"diagnosis":"Observed miss; cause is a hypothesis.","expected_prevention":"Catch the recorded contact."}
        edits=json.loads(t["nodes"][-1]["message"]["content"])
        t["nodes"][-1]["message"]["content"]=json.dumps({**explanation,**edits})
        t["calls"][0]["usage"]={"completion_tokens":100}
        report=t["info"]["repair_pipeline"]
        report.update(failure_case=case,baseline_replay=replay_failure(base,case),
                      candidate_replay=replay_failure(base,case),explanation=explanation,
                      accepted=False,selected_hash=fingerprint(base))
        report["gate"].update(seed_start=1060000,seed_end=1060511)
        t["rewards"]["improvement"]["score"]=0.
        return args

    def test_broad_gain_cannot_promote_when_specific_miss_remains(self):
        args=self.failure_fixture()
        result=control.validate_trace(*args)
        self.assertFalse(result[2]["accepted"])
        args[0]["info"]["repair_pipeline"]["accepted"]=True
        with self.assertRaises(ValueError): control.validate_trace(*args)

    def test_tools_are_allowed_only_in_new_protocol_and_within_budget(self):
        args=self.failure_fixture()
        args[1]["activity"].update(model_calls=2,tool_calls=1,tool_results=1)
        toolcall=copy.deepcopy(args[0]["calls"][0])
        toolcall["finish_reason"]="tool_calls"
        args[0]["calls"].insert(0,toolcall)
        args[0]["nodes"].insert(0,{"message":{"role":"assistant","content":"","tool_calls":[{"id":"test-tool","function":{"name":"bash","arguments":"{}"}}]}})
        args[0]["nodes"].insert(1,{"message":{"role":"tool","content":"Practice result"}})
        control.validate_trace(*args)
        args[1]["activity"]["tool_calls"]=25
        with self.assertRaises(ValueError): control.validate_trace(*args)

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
