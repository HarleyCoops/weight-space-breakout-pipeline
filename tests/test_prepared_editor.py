import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from weight_space_breakout.policy import faulty_policy, repair_edits
from weight_space_breakout.failures import failure_case
from weight_space_breakout.practice_tools import PracticeSession


class PracticeTests(unittest.TestCase):
    def test_ready_practice_and_explicit_submission(self):
        base=faulty_policy("reversed_controls")
        lab=PracticeSession(json.dumps(base.sparse()),failure_case(base,314159))
        self.assertTrue(lab.inspect()["ready"])
        self.assertFalse(lab.inspect()["baseline_replay"]["prevented_recorded_miss"])
        edits=repair_edits("reversed_controls")["edits"]
        result=lab.test(edits)
        self.assertTrue(result["practice_only"])
        self.assertNotIn("accepted",result)
        answer=json.loads(lab.submission("Hypothesis from trajectory","Expected behavioral change",edits))
        self.assertEqual(answer["edits"],edits)
        with self.assertRaises(ValueError): lab.submission("No change","No change",[])


@unittest.skipIf(sys.platform == "win32", "Verifiers v1 hosted adapter runs on Linux")
class PreparedAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_module_starts_real_mcp_server(self):
        import httpx
        from weight_space_breakout.taskset import BreakoutConfig, BreakoutTaskset
        from weight_space_breakout.editor_tools import EditorTaskConfig, EditorState
        task=BreakoutTaskset(BreakoutConfig(iterative=True,failure_driven=True,failure_seed=314159,
                            task=EditorTaskConfig(prepared_tools=True))).load()[0]
        state=EditorState().model_dump(mode="json")
        task_wire={"cls":"weight_space_breakout.taskset:BreakoutData","task":task.data.model_dump_json()}
        class Channel(BaseHTTPRequestHandler):
            def do_GET(self):
                payload=task_wire if self.path=="/task" else state
                body=json.dumps(payload).encode()
                self.send_response(200); self.end_headers(); self.wfile.write(body)
            def do_PUT(self):
                received=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                state.clear(); state.update(received)
                self.send_response(200); self.end_headers(); self.wfile.write(b"{}")
            def log_message(self,*args): pass
        channel=ThreadingHTTPServer(("127.0.0.1",0),Channel)
        worker=threading.Thread(target=channel.serve_forever,daemon=True)
        worker.start()
        with tempfile.TemporaryDirectory() as folder:
            port_file=Path(folder)/"port"
            env={k:v for k,v in os.environ.items() if not k.startswith("VF_")}
            env.update(MCP_PORT_FILE=str(port_file), MCP_HOST="127.0.0.1", MCP_PORT="0", VF_CONFIG="{}")
            env.update(VF_STATE_URL=f"http://127.0.0.1:{channel.server_port}/state",VF_STATE_SECRET="local-test")
            with (Path(folder)/"server.log").open("w+") as log:
                proc=subprocess.Popen([sys.executable,"-m","weight_space_breakout.editor_tools"],env=env,stdout=log,stderr=log)
                try:
                    deadline=time.monotonic()+20
                    while not port_file.exists() and proc.poll() is None and time.monotonic()<deadline:
                        await asyncio.sleep(.1)
                    log.flush()
                    self.assertTrue(port_file.exists(),(Path(folder)/"server.log").read_text())
                    async with httpx.AsyncClient() as client:
                        for _ in range(30):
                            try:
                                reply=await client.post(f"http://127.0.0.1:{port_file.read_text()}/mcp",headers={"Accept":"application/json, text/event-stream"},json={"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}})
                                reply.raise_for_status()
                                break
                            except (httpx.ConnectError,httpx.HTTPStatusError):
                                await asyncio.sleep(.1)
                        names={t["name"] for t in reply.json()["result"]["tools"]}
                        self.assertEqual(names,{"inspect_policy","test_edits","submit_edits"})
                        edits=repair_edits("reversed_controls")["edits"]
                        for tool,args in (("inspect_policy",{"tensor":"W3"}),
                                          ("test_edits",{"edits":edits}),
                                          ("submit_edits",{"diagnosis":"Observed direction error","expected_prevention":"Change direction","edits":edits})):
                            response=await client.post(f"http://127.0.0.1:{port_file.read_text()}/mcp",headers={"Accept":"application/json, text/event-stream"},json={"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":tool,"arguments":args}})
                            response.raise_for_status()
                            result=response.json()["result"]
                            self.assertFalse(result.get("isError",False),result)
                        self.assertEqual(json.loads(state["submission_json"])["edits"],edits)
                        self.assertTrue(state["submission_sha256"])
                finally:
                    proc.terminate()
                    await asyncio.to_thread(proc.wait,timeout=10)
                    channel.shutdown(); channel.server_close()
                    worker.join(timeout=5)

    async def test_preparation_tool_submission_and_stop(self):
        from weight_space_breakout.taskset import BreakoutConfig, BreakoutTaskset
        from weight_space_breakout.editor_tools import EditorState, EditorTaskConfig, RepairTools, ScalarEdit
        config=BreakoutConfig(iterative=True,failure_driven=True,failure_seed=314159,
                             task=EditorTaskConfig(prepared_tools=True))
        task=BreakoutTaskset(config).load()[0]
        trace=SimpleNamespace(state=EditorState(), info={}, num_turns=100)
        await task.setup(trace,None)
        self.assertTrue(trace.info["editor_setup"]["ready"])
        self.assertIsNone(task.data.max_editor_turns)
        self.assertFalse(await task.single_turn(trace))
        self.assertFalse(await task.edit_submitted(trace))
        tools=RepairTools(config.task.tools)
        await tools.setup_task(task.data)
        tools._inert_state=trace.state
        self.assertTrue((await tools.inspect_policy())["ready"])
        edits=[ScalarEdit(**e) for e in repair_edits("reversed_controls")["edits"]]
        self.assertTrue((await tools.test_edits(edits))["practice_only"])
        self.assertTrue((await tools.submit_edits("Trajectory suggests direction error","Changing direction should help",edits))["submitted"])
        self.assertTrue(await task.edit_submitted(trace))
        with self.assertRaises(ValueError): await tools.test_edits(edits)
        await task.improvement(trace)
        self.assertTrue(trace.info["repair_pipeline"]["valid"])
        self.assertEqual(trace.info["repair_pipeline"]["protocol"],"failure-driven-v2")


if __name__ == "__main__": unittest.main()
