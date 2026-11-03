"""Build a practice-only bundle for the editor's separate tool runtime."""
import inspect
import io
import zipfile
import json
import hashlib
import numpy as np
from . import game, policy, evaluation, failures


class PracticeSession:
    """Ready-to-use measurements, with only practice seeds exposed to the editor."""

    def __init__(self, snapshot, case):
        from .repair_loop import from_sparse, fingerprint
        self.base = from_sparse(snapshot)
        self.case = json.loads(case) if isinstance(case, str) else case
        if self.case["policy_hash"] != fingerprint(self.base):
            raise ValueError("Practice case belongs to a different policy")
        self.seeds = tuple(range(2040000, 2040032))
        self.before = [evaluation.rollout(self.base, s)["return"] for s in self.seeds]
        self.baseline_replay = failures.replay_failure(self.base, self.case)
        self.request_counts = {}
        self.best = None
        if self.baseline_replay["prevented_recorded_miss"] or not self.baseline_replay["same_ball_contact"]:
            raise ValueError("Practice setup did not reproduce the miss")

    def inspect(self, tensor=""):
        if tensor and tensor not in policy.SHAPES:
            raise ValueError("Unknown tensor; choose W1, b1, W2, b2, W3, or b3")
        return {"ready": True, "shapes": policy.SHAPES,
                "nonzero_weights": [e for e in self.base.sparse() if not tensor or e["tensor"] == tensor],
                "practice_games": len(self.seeds), "baseline_mean": float(np.mean(self.before)),
                "baseline_replay": self.baseline_replay,
                "assignment_limit": 64, "value_range": [-100,100],
                "best_practice_patch": self.best,
                "usage": "Call test_edits with scalar assignments, then submit_edits as your final action. No installation or file access is needed."}

    def test(self, edits):
        key = hashlib.sha256(json.dumps(edits,sort_keys=True,allow_nan=False).encode()).hexdigest()
        self.request_counts[key] = self.request_counts.get(key,0)+1
        repeated = self.request_counts[key]>1
        try:
            candidate = policy.apply_edits(self.base, {"edits": edits})
        except ValueError as error:
            return {"practice_only":True,"valid":False,"error":str(error),
                    "assignment_count":len(edits),"repeated_patch":repeated,
                    "next_step":"Correct the reported input error before retrying. Repeating this patch cannot change the result. Inspect the tensor shapes or submit a different valid tested patch."}
        after = [evaluation.rollout(candidate, s)["return"] for s in self.seeds]
        replay = failures.replay_failure(candidate, self.case)
        mean = float(np.mean(after))
        rank = (replay["prevented_recorded_miss"],mean)
        if self.best is None or rank > (self.best["prevented_recorded_miss"],self.best["candidate_mean"]):
            self.best = {"edits":edits,"prevented_recorded_miss":rank[0],"candidate_mean":mean}
        return {"practice_only": True, "valid":True, "target_replay": replay,
                "practice_games": len(self.seeds), "baseline_mean": float(np.mean(self.before)),
                "candidate_mean": mean, "practice_gain": float(np.mean(np.asarray(after)-self.before)),
                "repeated_patch":repeated,
                "next_step": ("This exact patch was already tested; repeating it gives no new evidence. Change the hypothesis or submit your best valid tested patch. " if repeated else "")+
                             "When ready, call submit_edits with your diagnosis, expected_prevention, and final assignments. inspect_policy can recover your best practice patch. A final submission is required even if you remain uncertain; independent verification can reject it."}

    def submission(self, diagnosis, expected_prevention, edits):
        text = json.dumps(dict(diagnosis=diagnosis, expected_prevention=expected_prevention, edits=edits), allow_nan=False)
        payload, _ = failures.parse_failure_submission(text)
        candidate = policy.apply_edits(self.base, payload)
        if not any(np.any(candidate.weights[k] != self.base.weights[k]) for k in policy.SHAPES):
            raise ValueError("Submit at least one actual weight change")
        return text


def practice_bundle():
    # Deliberately exclude gold policies, verifier gates, and selection/test seeds.
    modules = {
        "__init__.py":"",
        "game.py":inspect.getsource(game),
        "policy.py":"import json\nfrom pathlib import Path\nimport numpy as np\nSHAPES = "+repr(policy.SHAPES)+"\n"+
                    "\n".join(inspect.getsource(x) for x in (policy.Policy,policy.apply_edits,policy.parse_submission)),
        "evaluation.py":"import numpy as np\nfrom .game import Breakout\n"+inspect.getsource(evaluation.rollout),
        "failures.py":"import json\nimport numpy as np\nfrom .game import Breakout\n"+
                      "\n".join(inspect.getsource(x) for x in (failures.case_digest,failures.replay_failure)),
    }
    stream = io.BytesIO()
    with zipfile.ZipFile(stream,"w",zipfile.ZIP_DEFLATED) as archive:
        for name,source in modules.items():
            archive.writestr("repair_lab/"+name,source)
    return stream.getvalue()


PRACTICE_SCRIPT = '''"""Practice measurements only; the independent verifier makes promotion decisions."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"repair_lab.zip"))
from repair_lab.policy import Policy, SHAPES, apply_edits, parse_submission
from repair_lab.evaluation import rollout
from repair_lab.failures import replay_failure
base=apply_edits(Policy({k:np.zeros(s,np.float32) for k,s in SHAPES.items()}),
                 {"edits":json.loads((ROOT/"repair_policy.json").read_text())},max_edits=17253)
candidate=base
if len(sys.argv)>1:
    payload=parse_submission(Path(sys.argv[1]).read_text())
    candidate=apply_edits(base,{"edits":payload["edits"]})
case=json.loads((ROOT/"repair_failure.json").read_text())
seeds=range(1040000,1040032)
before=[rollout(base,s)["return"] for s in seeds]
after=[rollout(candidate,s)["return"] for s in seeds]
print(json.dumps({"practice_only":True,"target_replay":replay_failure(candidate,case),
                 "practice_games":32,"baseline_mean":float(np.mean(before)),
                 "candidate_mean":float(np.mean(after))},indent=2))
'''
