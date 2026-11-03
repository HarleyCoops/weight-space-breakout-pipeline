"""Verifiers v1 adapter: final JSON answer only; no submitted code execution."""
import asyncio
import base64
import json
from functools import lru_cache
import verifiers.v1 as vf
from .evaluation import PRACTICE_SEEDS, TEST_SEEDS, evaluate, score_submission
from .policy import FAULTS, faulty_policy, repair_edits
from .repair_loop import from_sparse, fingerprint, practice_feedback, verify_round
from .failures import find_failure, parse_failure_submission, verify_failure_round
from .editor_tools import EditorState, EditorTaskConfig, RepairTools

RULES = """Independent Breakout v1. Rows/columns 0..9; outer ring is walls.
Paddle is three cells on row 8, moves before ball. Left/right move one cell, center clamped
to columns 2..7. No-op/up/down do nothing. Ball starts row 6, column 1..8,
moving up and diagonally left or right. Bricks occupy rows 1..3, columns 1..8.
Ball moves diagonally one cell per tick, reflects at side/top walls. Vertical
and horizontal brick neighbors reflect the corresponding velocity component;
otherwise a diagonal brick reflects vertical velocity. Hit bricks disappear.
Recheck walls after reflections; if destination is still a brick, ball stays
in place that tick. At row 8, any of the three paddle cells bounces upward; otherwise
episode ends. +1 per brick, -1 on miss. Clear all 24 bricks or stop at 200 ticks.
Observation: 10x10x5 binary, flattened row/column/channel; channels agent,
ball, bricks, walls, previous agent-or-ball. Previous channel initially zero.
Policy: h1=tanh(x@W1+b1); h2=tanh(h1@W2+b2); logits=h2@W3+b3.
Shapes: W1 500x32, b1 32, W2 32x32, b2 32, W3 32x5, b3 5.
Actions sampled from softmax: 0 no-op, 1 up, 2 down, 3 left, 4 right.
This is a handwritten teaching policy with a deliberate fault, NOT PPO.
Repair its behavior through at most 64 scalar assignments, values [-100,100].
Return ONLY JSON: {"edits":[{"tensor":"W3","index":[0,3],"value":-1.0}]}.
No gradients, training, simulation, or tool calls are allowed in this one-shot task.
Score = clip((edited return - baseline return) / max(1, reference return -
baseline return), 0, 1) on separate evaluation seeds. Reference is an intact
handwritten ball-column tracker. All unlisted weights below are exactly zero.
"""


class BreakoutData(vf.TaskData):
    fault: str
    loop_policy: str = ""
    round_index: int = 0
    final_round: bool = False
    failure_case_json: str = ""
    max_editor_turns: int | None = 1
    protocol_id: str = ""


class BreakoutConfig(vf.TasksetConfig):
    task: EditorTaskConfig = EditorTaskConfig()
    iterative: bool = False
    policy_snapshot: str = ""
    policy_snapshot_b64: str = ""
    policy_sha256: str = ""
    round_index: int = 0
    final_round: bool = False
    previous_feedback: str = ""
    previous_feedback_b64: str = ""
    failure_driven: bool = False
    failure_seed: int = -1
    max_editor_turns: int = 8


@lru_cache(maxsize=3)
def prompt_for(fault):
    p = faulty_policy(fault)
    result = evaluate(p, PRACTICE_SEEDS)
    return RULES + "\nPractice mean return: " + str(result["mean_return"]) + "\nNonzero weights:\n" + json.dumps(p.sparse())


class BreakoutTask(vf.Task[BreakoutData, EditorState, EditorTaskConfig]):
    @classmethod
    def toolsets(cls, config):
        return [RepairTools(config.tools)] if config.prepared_tools else []

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        if self.data.protocol_id == "failure-driven-v2":
            from .practice_tools import PracticeSession
            import numpy as np
            lab = await asyncio.to_thread(PracticeSession, self.data.loop_policy, self.data.failure_case_json)
            trace.info["editor_setup"] = {"ready": True, "numpy_version": np.__version__,
                "practice_games": len(lab.seeds), "baseline_mean": float(np.mean(lab.before)),
                "baseline_replay": lab.baseline_replay, "tools": ["repair_inspect_policy", "repair_test_edits", "repair_submit_edits"],
                "max_editor_turns": None, "final_action": "repair_submit_edits"}
            return
        if self.data.failure_case_json:
            from .practice_tools import practice_bundle, PRACTICE_SCRIPT
            for name,content in {"repair_lab.zip":practice_bundle(),
                                 "repair_practice.py":PRACTICE_SCRIPT.encode(),
                                 "repair_policy.json":self.data.loop_policy.encode(),
                                 "repair_failure.json":self.data.failure_case_json.encode()}.items():
                await runtime.write("/tmp/"+name,content)

    @vf.stop
    async def edit_submitted(self, trace: vf.Trace) -> bool:
        return self.data.protocol_id == "failure-driven-v2" and bool(trace.state.submission_json)

    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        if self.data.protocol_id == "failure-driven-v2":
            return False
        if self.data.failure_case_json:
            if trace.num_turns >= self.data.max_editor_turns:
                return True
            messages = getattr(trace,"assistant_messages",[])
            if messages and getattr(messages[-1],"tool_calls",None):
                return False
            try:
                parse_failure_submission(trace.last_reply or "")
                return True
            except (ValueError,TypeError,OverflowError,RecursionError):
                return False
        return trace.num_turns >= 1

    @vf.reward
    async def improvement(self, trace: vf.Trace) -> float:
        if self.data.loop_policy:
            if self.data.failure_case_json:
                prepared = self.data.protocol_id == "failure-driven-v2"
                answer = trace.state.submission_json if prepared else (trace.last_reply or "")
                report = await asyncio.to_thread(verify_failure_round, answer, from_sparse(self.data.loop_policy), self.data.round_index, json.loads(self.data.failure_case_json), self.data.final_round, "failure-driven-v2" if prepared else "failure-driven-v1")
                if prepared:
                    report["submission"] = {"source":"repair_submit_edits", "json":answer,
                                            "sha256":trace.state.submission_sha256}
            else:
                report = await asyncio.to_thread(verify_round, trace.last_reply or "", from_sparse(self.data.loop_policy), self.data.round_index, self.data.final_round)
            trace.info["repair_pipeline"] = report
            return report["reward"]
        report = await asyncio.to_thread(score_submission, trace.last_reply or "", self.data.fault, TEST_SEEDS)
        trace.info["weight_repair"] = report
        return report["reward"]

    async def validate(self, runtime: vf.Runtime) -> bool:
        if self.data.loop_policy:
            # Iterative snapshots need not have a known improving solution.
            return fingerprint(from_sparse(self.data.loop_policy)) != ""
        report = await asyncio.to_thread(score_submission, json.dumps(repair_edits(self.data.fault)), self.data.fault, TEST_SEEDS)
        return report["valid"] and abs(report["reward"] - 1) < 1e-6


class BreakoutTaskset(vf.Taskset[BreakoutTask, BreakoutConfig]):
    def load(self) -> list[BreakoutTask]:
        if self.config.iterative:
            snapshot_text = base64.b64decode(self.config.policy_snapshot_b64, validate=True).decode() if self.config.policy_snapshot_b64 else self.config.policy_snapshot
            previous_feedback = base64.b64decode(self.config.previous_feedback_b64, validate=True).decode() if self.config.previous_feedback_b64 else self.config.previous_feedback
            if not 0 <= self.config.round_index < 20 or len(previous_feedback) > 20000:
                raise ValueError("Invalid bounded repair round")
            policy = from_sparse(snapshot_text) if snapshot_text else faulty_policy("reversed_controls")
            if self.config.policy_sha256 and fingerprint(policy) != self.config.policy_sha256:
                raise ValueError("Policy snapshot does not match pinned hash")
            snapshot = json.dumps(policy.sparse())
            if self.config.failure_driven:
                prepared = self.config.task.prepared_tools
                if (not prepared and not 1 <= self.config.max_editor_turns <= 8) or not -1 <= self.config.failure_seed < 2**32:
                    raise ValueError("Invalid tool budget or failure seed")
                case = find_failure(policy,self.config.round_index,None if self.config.failure_seed == -1 else self.config.failure_seed,
                                    2040000 if prepared else 1040000)
                prompt = RULES.split("This is a handwritten")[0]
                prompt += "\nThe CURRENT policy failed in the recorded episode below. The cause is unknown. Diagnose the observed behavior, propose up to 64 scalar weight assignments in [-100,100], and explain the concrete behavioral change you expect to prevent this miss. Do not assume a particular planted fault or that a single bad weight exists. A sampled action can fail even when it is not the most likely action. Earlier decisions, missing features, and limited representation are also possibilities. Your explanation is a hypothesis, not a verified causal conclusion.\n"
                if prepared:
                    prompt += "Tool calling is enabled and required. The practice tools and NumPy are already prepared; do not install anything or explore files. There is no model-turn or total-output-token limit in this protocol. Use repair_inspect_policy to inspect weights and baseline results, and repair_test_edits to test a COMPLETE proposed patch on the recorded failure and 32 practice games. Each practice call starts from the current baseline. Practice scores are not the independent promotion test. No gradient training. Keep tool calls and explanations focused.\n"
                    prompt += "Your FINAL action MUST be repair_submit_edits with diagnosis, expected_prevention, and your complete edits. This commits the actual proposal and ends the attempt. Use practice feedback to revise your hypothesis; when you have a concrete proposal, submit it. Do not finish with analysis, a plan, or a text-only answer. A submitted proposal may fail verification; submit your best tested change rather than waiting for certainty.\n"
                    prompt += "A patch contains only assignments you want to change, not a copy of the full network. Count the assignments before calling a tool: at most 64 TOTAL per patch. Omit unchanged entries. Calls start from the baseline, so splitting a large patch across calls does not accumulate changes. Fix validation errors before retrying. Repeating an identical patch cannot produce new evidence because the practice games use fixed randomness. If experiments stop providing useful new evidence, finish with your best valid tested proposal, even if it may fail the independent verifier.\n"
                else:
                    prompt += "Tools and practice simulations are allowed. No gradient training. The bash runtime contains /tmp/repair_policy.json, /tmp/repair_failure.json, /tmp/repair_lab.zip, and /tmp/repair_practice.py. Install numpy==2.4.1 if needed. Run python /tmp/repair_practice.py for the baseline, or pass a candidate JSON file path to test an edit. These are practice measurements only; modifying the helper cannot change the independent verifier. You have at most " + str(self.config.max_editor_turns) + " model turns including the final answer.\n"
                    prompt += 'Return final JSON with exactly these fields: {"diagnosis":"brief explanation tied to observed steps","expected_prevention":"what will change and why it should catch this ball","edits":[{"tensor":"tensor name","index":[0],"value":0}]}.\n'
                prompt += "The verifier freezes the episode history before intervention_step and reuses the same random draws, then asks whether the edited policy catches the SAME ball contact that was missed. Passing that replay alone is insufficient: a candidate must also pass a separate broad improvement check. Full-episode replay from reset and conditional replay after intervention are different tests.\n"
                prompt += "Current policy hash: " + fingerprint(policy)
                prompt += "\nObserved failure: " + json.dumps(case,separators=(",",":"))
                prompt += "\nPrevious attempt summary: " + (previous_feedback or "None; first round")
                prompt += "\nCurrent nonzero weights (all others zero):\n" + snapshot
                return [BreakoutTask(BreakoutData(id=f"failure-{self.config.round_index}",prompt=prompt,
                        fault="observed_failure",loop_policy=snapshot,round_index=self.config.round_index,
                        final_round=self.config.final_round,failure_case_json=json.dumps(case),
                        max_editor_turns=None if prepared else self.config.max_editor_turns,
                        protocol_id="failure-driven-v2" if prepared else "failure-driven-v1"),self.config.task)]
            rules = RULES.split("Score =")[0].replace("in this one-shot task", "during your single response in this iterative experiment")
            prompt = rules + "\nThis is one round of an iterative repair loop. Improve the CURRENT policy below. You may use the supplied practice feedback and previous attempt summary. Do not call tools or run simulations during this response. A verifier tests your candidate on fresh seeds after you submit it; a non-improving edit is rejected. All unlisted weights are exactly zero.\n"
            prompt += "Current policy hash: " + fingerprint(policy)
            prompt += "\nPractice feedback: " + json.dumps(practice_feedback(policy, self.config.round_index))
            prompt += "\nPrevious attempt summary: " + (previous_feedback or "None; first round")
            prompt += "\nCurrent nonzero weights:\n" + snapshot
            return [BreakoutTask(BreakoutData(id=f"iterative-{self.config.round_index}", prompt=prompt,
                    fault="reversed_controls", loop_policy=snapshot, round_index=self.config.round_index,
                    final_round=self.config.final_round), self.config.task)]
        return [BreakoutTask(BreakoutData(id=f"repair-{i}", prompt=prompt_for(fault), fault=fault), self.config.task)
                for i, fault in enumerate(FAULTS)]
