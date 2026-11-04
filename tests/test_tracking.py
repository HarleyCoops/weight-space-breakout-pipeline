import json
from pathlib import Path
import tempfile
import unittest
from pipeline.tracking import snapshots


class TrackingTests(unittest.TestCase):
    def fixture(self,folder):
        def save(name,value):
            path=folder/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value),encoding="utf-8")
        save("plan.json",{"model_repo":"owner/policy"})
        save("state.json",{})
        save("round-00/submission.json",{"evaluation_id":"eval-1","round_index":0,"baseline_hash":"baseline"})
        return save

    def test_service_failure_never_becomes_zero_game_reward(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);save=self.fixture(folder)
            self.assertEqual(list(snapshots(folder)),[])
            save("round-00/execution-failure.json",{"trace_available":False,"promoted":False})
            item=list(snapshots(folder))[0]
            self.assertEqual(item["metrics"]["execution/failed"],1)
            self.assertEqual(item["selected_hash"],"baseline")
            self.assertFalse(any(k.startswith(("gate/","audit/","qualification/")) for k in item["metrics"]))

    def test_practice_gate_audit_and_selection_remain_distinct(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);save=self.fixture(folder)
            save("round-00/verifier-report.json",{"valid":True,"accepted":False,"changed_parameters":3,
                "selected_hash":"baseline","gate":{"gain":-1.,"candidate_return":2.,"baseline_returns":[0,2,4],"candidate_returns":[1,1,4]},
                "final_audit":{"gain":0.,"candidate_return":3.}})
            save("round-00/trace.json",{"nodes":[{"message":{"role":"tool","name":"repair_propose_edits",
                "content":json.dumps({"valid":True,"candidate_mean":20.})}}]})
            item=list(snapshots(folder))[0]
            self.assertEqual(item["practice"][0]["practice/candidate_mean"],20.)
            self.assertEqual(item["metrics"]["gate/gain"],-1.)
            self.assertEqual(item["metrics"]["audit/gain"],0.)
            self.assertEqual(item["metrics"]["selection/accepted"],0)
            self.assertEqual(item["metrics"]["execution/failed"],0)
            self.assertEqual([item["metrics"]["gate/"+k] for k in ("wins","losses","ties","worst_paired_change")],[1,1,1,-1])
            self.assertNotIn("trace.json",[p.name for p in item["files"]])
            self.assertEqual(item["id"],list(snapshots(folder))[0]["id"])

    def test_new_publication_updates_snapshot_without_creating_another_attempt(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);save=self.fixture(folder)
            save("round-00/verifier-report.json",{"valid":True,"accepted":True,"selected_hash":"repaired"})
            before=list(snapshots(folder))[0]
            save("round-00/publication.json",{"revision":"pinned-revision"})
            after=list(snapshots(folder))[0]
            self.assertEqual(before["id"],after["id"])
            self.assertNotEqual(before["data_sha256"],after["data_sha256"])


if __name__=="__main__": unittest.main()
