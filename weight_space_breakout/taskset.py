"""Verifiers v1 adapter: final JSON answer only; no submitted code execution."""
import asyncio
import base64
import json
from functools import lru_cache
import verifiers.v1 as vf
from .evaluation import PRACTICE_SEEDS, TEST_SEEDS, evaluate, score_submission
from .policy import FAULTS, faulty_policy, repair_edits
from .repair_loop import from_sparse, fingerprint, practice_feedback, verify_round

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


class BreakoutConfig(vf.TasksetConfig):
    iterative: bool = False
    policy_snapshot: str = ""
    policy_snapshot_b64: str = ""
    policy_sha256: str = ""
    round_index: int = 0
    final_round: bool = False
    previous_feedback: str = ""
    previous_feedback_b64: str = ""


@lru_cache(maxsize=3)
def prompt_for(fault):
    p = faulty_policy(fault)
    result = evaluate(p, PRACTICE_SEEDS)
    return RULES + "\nPractice mean return: " + str(result["mean_return"]) + "\nNonzero weights:\n" + json.dumps(p.sparse())


class BreakoutTask(vf.Task[BreakoutData]):
    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward
    async def improvement(self, trace: vf.Trace) -> float:
        if self.data.loop_policy:
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
