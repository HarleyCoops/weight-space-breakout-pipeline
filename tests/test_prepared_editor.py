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
    def test_basic_feedback_keeps_replay_diagnostic(self):
        base=faulty_policy("reversed_controls")
        lab=PracticeSession(json.dumps(base.sparse()),failure_case(base,314159),"basic-repair-v1")
        result=lab.test(repair_edits("reversed_controls")["edits"])
        self.assertFalse(result["target_replay_required"])
        self.assertIn("diagnostic only",result["objective"])
        self.assertEqual(lab.seeds,tuple(range(3040000,3040032)))

    def test_invalid_and_repeated_patch_feedback(self):
        base=faulty_policy("reversed_controls")
        lab=PracticeSession(json.dumps(base.sparse()),failure_case(base,314159))
        invalid=[{"tensor":"W3","index":[32,0],"value":1.0}]
        first=lab.test(invalid)
        self.assertFalse(first["valid"])
        self.assertIn("W3",first["error"])
        self.assertIn("shape",first["error"])
        self.assertTrue(lab.test(invalid)["repeated_patch"])
        oversize=[{"tensor":"W1","index":[i,0],"value":1.0} for i in range(69)]
        self.assertIn("69",lab.test(oversize)["error"])
        edits=repair_edits("reversed_controls")["edits"]
        first=lab.test(edits); repeated=lab.test(edits)
        self.assertFalse(first["repeated_patch"])
        self.assertTrue(repeated["repeated_patch"])
        self.assertEqual(first["candidate_mean"],repeated["candidate_mean"])
        self.assertEqual(lab.inspect()["best_practice_patch"]["edits"],edits)
        self.assertNotIn("accepted",repeated)

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
    async def test_full_agent_stops_after_committed_edit(self):
        await self.run_scripted_agent(False)

    async def test_basic_agent_promotes_broad_repair_without_target_rescue(self):
        await self.run_scripted_agent(True)

    async def test_qualified_agent_ends_on_first_qualifying_proposal(self):
        await self.run_scripted_agent(True,qualified=True)

    async def run_scripted_agent(self,basic,qualified=False):
        """Scripted model, real harness/MCP/state/stop/scoring; no paid inference."""
        import verifiers.v1 as vf
        from verifiers.v1.configs.client import EvalClientConfig
        from verifiers.v1.configs.agent import TimeoutConfig
        from weight_space_breakout.taskset import BreakoutConfig, BreakoutTaskset
        from weight_space_breakout.editor_tools import EditorTaskConfig
        task=BreakoutTaskset(BreakoutConfig(iterative=True,failure_driven=True,failure_seed=314159,basic_repair=basic,
                            final_round=qualified,
                            task=EditorTaskConfig(prepared_tools=True,qualified_proposals=qualified))).load()[0]
        edits=repair_edits("reversed_controls")["edits"]
        script=[("repair_inspect_policy",{"tensor":"W3"}),
                ("repair_test_edits",{"edits":edits}),
                ("repair_submit_edits",{"diagnosis":"Scripted lifecycle fixture",
                  "expected_prevention":"Correct reversed direction","edits":edits})]
        if qualified:
            # Test fixture only, never supplied to the hosted editor.
            edits=edits+[{"tensor":"b3","index":[4],"value":2.0}]
            # An inert edit to a zero outgoing neuron changes weights but not play.
            script=[("repair_inspect_policy",{"tensor":"W3"}),
                    ("repair_propose_edits",{"diagnosis":"Nonqualifying lifecycle fixture",
                     "expected_prevention":"Check that zero gain leaves editing open",
                     "edits":[{"tensor":"b2","index":[31],"value":1.0}]}),
                    ("repair_propose_edits",{"diagnosis":"Qualifying lifecycle fixture",
                     "expected_prevention":"Correct reversed direction","edits":edits})]
        requests=[]
        class Provider(BaseHTTPRequestHandler):
            def do_POST(self):
                request=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                index=len(requests); requests.append(request)
                if index>=len(script):
                    self.send_response(500); self.end_headers(); self.wfile.write(b"unexpected extra model call")
                    return
                name,args=script[index]
                call={"id":f"call_{index}","type":"function","function":{"name":name,"arguments":json.dumps(args)}}
                response={"id":f"fixture_{index}","object":"chat.completion","created":0,
                          "model":"scripted-fixture","choices":[{"index":0,"finish_reason":"tool_calls",
                          "message":{"role":"assistant","content":None,"tool_calls":[call]}}],
                          "usage":{"prompt_tokens":10,"completion_tokens":10,"total_tokens":20}}
                if request.get("stream"):
                    response["object"]="chat.completion.chunk"
                    choice=response["choices"][0]
                    choice["delta"]=choice.pop("message")
                    choice["delta"]["tool_calls"][0]["index"]=0
                    body=("data: "+json.dumps(response)+"\n\ndata: [DONE]\n\n").encode()
                    mime="text/event-stream"
                else:
                    body=json.dumps(response).encode(); mime="application/json"
                self.send_response(200); self.send_header("Content-Type",mime)
                self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)
            def log_message(self,*args): pass
        provider=ThreadingHTTPServer(("127.0.0.1",0),Provider)
        worker=threading.Thread(target=provider.serve_forever,daemon=True); worker.start()
        config=vf.AgentConfig(model="scripted-fixture",harness=vf.HarnessConfig(id="null"),
                              runtime=vf.SubprocessConfig(),
                              client=EvalClientConfig(base_url=f"http://127.0.0.1:{provider.server_port}/v1",api_key_var="LIFECYCLE_TEST_KEY"),
                              sampling=vf.SamplingConfig(tool_choice="required",parallel_tool_calls=False),
                              timeout=TimeoutConfig(setup=60,rollout=60,finalize=30,scoring=60))
        try:
            async with vf.Agent(config) as agent:
                trace=await agent.run(task)
            self.assertTrue(trace.ok,trace.model_dump_json())
            self.assertEqual(trace.stop_condition,"edit_submitted")
            self.assertEqual(len(requests),3)
            self.assertEqual(trace.num_turns,3)
            self.assertEqual(json.loads(trace.state.submission_json)["edits"],edits)
            self.assertTrue(trace.info["repair_pipeline"]["valid"])
            report=trace.info["repair_pipeline"]
            self.assertEqual(report["protocol"],"qualified-repair-v1" if qualified else ("basic-repair-v1" if basic else "failure-driven-v2"))
            self.assertEqual(report["target_replay_required"],not basic)
            self.assertEqual(report["candidate_replay"]["prevented_recorded_miss"],qualified)
            self.assertEqual(report["accepted"],basic)
            if qualified:
                self.assertEqual(trace.state.valid_proposals,2)
                self.assertEqual(report["submission"]["source"],"repair_propose_edits")
                self.assertEqual(report["practice_qualification"]["seed_start"],4040000)
                self.assertEqual(report["gate"]["seed_start"],4060000)
                self.assertEqual(report["final_audit"]["seed_start"],4090000)
                names={t["function"]["name"] for t in requests[0]["tools"]}
                self.assertEqual(names,{"repair_inspect_policy","repair_propose_edits"})
                from pipeline.control import check_first_qualification
                wire=trace.model_dump(mode="json")
                last=next(n["message"] for n in reversed(wire["nodes"]) if n["message"]["role"]=="assistant")
                check_first_qualification(wire,report["submission"],last["tool_calls"][0])
            self.assertTrue(all(r["tool_choice"]=="required" and not r["parallel_tool_calls"] for r in requests))
        finally:
            provider.shutdown(); provider.server_close(); worker.join(timeout=5)

    async def test_qualification_commits_once_and_keeps_nonqualifying_edit_open(self):
        from weight_space_breakout.taskset import BreakoutConfig,BreakoutTaskset
        from weight_space_breakout.editor_tools import EditorTaskConfig,EditorState,ScalarEdit
        from weight_space_breakout.qualified_tools import QualifiedRepairTools
        config=BreakoutConfig(iterative=True,failure_driven=True,basic_repair=True,failure_seed=314159,
                             task=EditorTaskConfig(prepared_tools=True,qualified_proposals=True))
        task=BreakoutTaskset(config).load()[0]
        tools=QualifiedRepairTools(config.task.tools)
        await tools.setup_task(task.data)
        tools._inert_state=EditorState()
        inert=[ScalarEdit(tensor="b2",index=[31],value=1.0)]
        first=await tools.propose_edits("Test inert feature","No expected return gain",inert)
        self.assertFalse(first["submitted"])
        self.assertFalse(tools.state.submission_json)
        edits=[ScalarEdit(**e) for e in repair_edits("reversed_controls")["edits"]]
        edits.append(ScalarEdit(tensor="b3",index=[4],value=2.0))
        result=await tools.propose_edits("Direction hypothesis","Direction correction should help",edits)
        self.assertTrue(result["submitted"])
        self.assertNotIn("accepted",result)
        receipt=tools.state.submission_json
        with self.assertRaises(ValueError):
            await tools.propose_edits("Another hypothesis","Try another change",inert)
        self.assertEqual(tools.state.submission_json,receipt)

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
                        for definition in reply.json()["result"]["tools"]:
                            if definition["name"] in ("test_edits","submit_edits"):
                                self.assertEqual(definition["inputSchema"]["properties"]["edits"]["maxItems"],64)
                        invalid=[{"tensor":"W1","index":[i,0],"value":1.0} for i in range(69)]
                        response=await client.post(f"http://127.0.0.1:{port_file.read_text()}/mcp",headers={"Accept":"application/json, text/event-stream"},json={"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"test_edits","arguments":{"edits":invalid}}})
                        result=response.json()["result"]
                        self.assertTrue(result["isError"],result)
                        self.assertIn("64",json.dumps(result))
                        self.assertFalse(state["submission_json"])
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
