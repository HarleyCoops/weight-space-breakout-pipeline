"""Build a practice-only bundle for the editor's separate tool runtime."""
import inspect
import io
import zipfile
from . import game, policy, evaluation, failures


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
