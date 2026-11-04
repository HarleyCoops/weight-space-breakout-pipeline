"""First practice-qualified proposal is frozen for independent verification."""
import asyncio
import hashlib
import verifiers.v1 as vf
from .editor_tools import EditorState, EditList
from .practice_tools import PracticeSession


class QualifiedRepairTools(vf.Toolset[vf.ToolsetConfig, EditorState]):
    """First qualifying proposal ends exploration; promotion remains independent."""
    TOOL_PREFIX = "repair"

    async def setup_task(self, task):
        if task.protocol_id != "qualified-repair-v1":
            raise ValueError("Wrong protocol for automatic qualification")
        self.lab = await asyncio.to_thread(PracticeSession, task.loop_policy, task.failure_case_json, task.protocol_id)

    def require_open(self):
        if self.state.submission_json:
            raise ValueError("A qualifying proposal is already frozen; editing has ended")

    @vf.tool
    async def inspect_policy(self, tensor: str = "") -> dict:
        """Inspect current weights and baseline practice results. Optional tensor W1, b1, W2, b2, W3, or b3. Use propose_edits to propose changes."""
        self.require_open()
        return self.lab.inspect(tensor)

    @vf.tool
    async def propose_edits(self, diagnosis: str, expected_prevention: str, edits: EditList) -> dict:
        """Propose a complete weight edit with a brief diagnosis and expected improvement. Each proposal starts from the baseline. It qualifies when 32 practice games show mean gain>0.10 and gain>2*paired_SE, with a real weight change. The FIRST qualifying proposal is frozen automatically and ends editing; a separate independent verifier decides promotion. The recorded miss is diagnostic only. At most 64 scalar assignments; omit unchanged weights."""
        self.require_open()
        payload = [e.model_dump() for e in edits]
        committed = self.lab.submission(diagnosis, expected_prevention, payload)
        result = await asyncio.to_thread(self.lab.test, payload)
        self.state.valid_proposals += 1
        result["valid_proposal_number"] = self.state.valid_proposals
        result["submitted"] = False
        if result["practice_qualified"]:
            self.state.submission_json = committed
            self.state.submission_sha256 = hashlib.sha256(committed.encode()).hexdigest()
            result.update(submitted=True, submission_sha256=self.state.submission_sha256,
                          next_step="First qualifying edit frozen. Editing has ended; independent verification follows. This does not yet mean the edit was accepted or published.")
        return result


if __name__ == "__main__":
    QualifiedRepairTools.run()
