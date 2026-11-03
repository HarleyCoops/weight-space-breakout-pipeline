"""Task-scoped prepared practice tools and an explicit terminal edit submission."""
import asyncio
import hashlib
import verifiers.v1 as vf
from pydantic import BaseModel, ConfigDict
from .practice_tools import PracticeSession


class ScalarEdit(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    tensor: str
    index: list[int]
    value: float


class EditorState(vf.State):
    submission_json: str = ""
    submission_sha256: str = ""


class EditorTaskConfig(vf.TaskConfig):
    prepared_tools: bool = False
    tools: vf.ToolsetConfig = vf.ToolsetConfig()


class RepairTools(vf.Toolset[vf.ToolsetConfig, EditorState]):
    TOOL_PREFIX = "repair"

    async def setup_task(self, task):
        self.lab = await asyncio.to_thread(PracticeSession, task.loop_policy, task.failure_case_json)

    def require_open(self):
        if self.state.submission_json:
            raise ValueError("The edit has already been submitted; this attempt is complete")

    @vf.tool
    async def inspect_policy(self, tensor: str = "") -> dict:
        """Inspect current nonzero weights and the already-measured baseline. Optional tensor: W1, b1, W2, b2, W3, b3. Tools are prepared; no installation is needed."""
        self.require_open()
        return self.lab.inspect(tensor)

    @vf.tool
    async def test_edits(self, edits: list[ScalarEdit]) -> dict:
        """Test up to 64 scalar assignments on the recorded miss and 32 practice games. Each call starts from the current policy; include the complete proposed patch. Practice results do not decide promotion."""
        self.require_open()
        return await asyncio.to_thread(self.lab.test, [e.model_dump() for e in edits])

    @vf.tool
    async def submit_edits(self, diagnosis: str, expected_prevention: str, edits: list[ScalarEdit]) -> dict:
        """FINAL ACTION: commit your complete weight edit and finish this attempt. Explain the observed failure and predicted prevention. Up to 64 assignments in [-100,100]; at least one real change. The independent verifier evaluates it after submission. Do not call another tool after this."""
        self.require_open()
        text = self.lab.submission(diagnosis, expected_prevention, [e.model_dump() for e in edits])
        self.state.submission_json = text
        self.state.submission_sha256 = hashlib.sha256(text.encode()).hexdigest()
        return {"submitted": True, "submission_sha256": self.state.submission_sha256,
                "message": "Final edit recorded. This attempt is complete; independent verification follows."}
