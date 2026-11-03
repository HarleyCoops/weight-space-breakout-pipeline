"""Task-scoped prepared practice tools and an explicit terminal edit submission."""
import asyncio
import hashlib
from typing import Annotated, Literal
import verifiers.v1 as vf
from pydantic import BaseModel, ConfigDict, Field
from .practice_tools import PracticeSession


class ScalarEdit(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    tensor: Literal["W1", "b1", "W2", "b2", "W3", "b3"]
    index: Annotated[list[Annotated[int, Field(ge=0)]], Field(min_length=1, max_length=2)]
    value: Annotated[float, Field(ge=-100, le=100, allow_inf_nan=False)]


EditList = Annotated[list[ScalarEdit], Field(min_length=1, max_length=64,
    description="1 to 64 scalar assignments total. Omit unchanged weights. Each call is a complete patch relative to the baseline, not the full network.")]


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
    async def test_edits(self, edits: EditList) -> dict:
        """Test up to 64 scalar assignments on the recorded miss and 32 practice games. Each call starts from the current policy; include the complete proposed patch. Practice results do not decide promotion."""
        self.require_open()
        return await asyncio.to_thread(self.lab.test, [e.model_dump() for e in edits])

    @vf.tool
    async def submit_edits(self, diagnosis: str, expected_prevention: str, edits: EditList) -> dict:
        """FINAL ACTION: commit your complete weight edit and finish this attempt. Explain the observed failure and predicted prevention. Up to 64 assignments in [-100,100]; at least one real change. The independent verifier evaluates it after submission. Do not call another tool after this."""
        self.require_open()
        text = self.lab.submission(diagnosis, expected_prevention, [e.model_dump() for e in edits])
        self.state.submission_json = text
        self.state.submission_sha256 = hashlib.sha256(text.encode()).hexdigest()
        return {"submitted": True, "submission_sha256": self.state.submission_sha256,
                "message": "Final edit recorded. This attempt is complete; independent verification follows."}


if __name__ == "__main__":
    RepairTools.run()
