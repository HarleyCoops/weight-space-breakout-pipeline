"""Bounded hosted repair controller. No local LLM or game evaluation."""
from __future__ import annotations
import argparse
import base64
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from weight_space_breakout import __version__
from weight_space_breakout.policy import faulty_policy, tracking_policy, apply_edits, parse_submission, SHAPES
from weight_space_breakout.repair_loop import fingerprint, from_sparse, passes_gate
from weight_space_breakout.failures import parse_failure_submission, case_digest
from pipeline.failure_audit import summarize_failure

RUN = Path(os.environ.get("REPAIR_RUN_DIR", ROOT / "artifacts/iterative-repair"))
MODEL_ID = os.environ.get("REPAIR_MODEL_ID", "HarleyCooper/weight-space-breakout-iterative")
SPACE_ID = os.environ.get("REPAIR_SPACE_ID", "HarleyCooper/weight-space-breakout-repair-loop")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def current_policy(state):
    policy = from_sparse(json.dumps(read(RUN / "current-policy.json")))
    if fingerprint(policy) == state["current_hash"]:
        return policy
    # Recover a process interruption between committing state and its derived snapshot.
    if state["history"]:
        last = state["history"][-1]
        if fingerprint(policy) == last["baseline_hash"]:
            folder = RUN / f"round-{last['round']:02d}"
            name = "candidate-policy.json" if last["accepted"] else "baseline-policy.json"
            recovered = from_sparse(json.dumps(read(folder / name)))
            if fingerprint(recovered) == state["current_hash"] == last["selected_hash"]:
                save(RUN / "current-policy.json", recovered.sparse())
                return recovered
    raise ValueError("Incumbent changed outside the controller")


def prime(args, timeout=120):
    executable = os.environ.get("PRIME_CLI", "/home/chris/.local/bin/prime" if os.name == "nt" else "prime")
    command = ["wsl", "-d", "Ubuntu", "--", executable] if os.name == "nt" else [executable]
    result = subprocess.run(command + args, capture_output=True, timeout=timeout)
    text = result.stdout.decode("utf-8", errors="replace")
    if result.returncode:
        # No credentials are passed in command arguments or written into run records.
        raise RuntimeError("Prime command failed: " + result.stderr.decode("utf-8", errors="replace")[-2000:] + text[-2000:])
    return text


def json_output(text):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
            return value
        except ValueError:
            pass
    raise ValueError("No JSON in Prime response")


def initialize():
    if (RUN / "plan.json").exists():
        return read(RUN / "plan.json")
    protocol = os.environ.get("REPAIR_PROTOCOL", "failure-driven-v2")
    if protocol not in ("failure-driven-v2", "basic-repair-v1", "qualified-repair-v1"):
        raise ValueError("Unknown new-run protocol")
    qualified = protocol == "qualified-repair-v1"
    basic = protocol in ("basic-repair-v1", "qualified-repair-v1")
    plan = {"created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "rounds": 1 if basic else 3,
            "environment": f"harleycooper/weight-space-breakout@{__version__}",
            "editor": "qwen/qwen3.6-35b-a3b", "max_tokens": 16384, "temperature": 0.6 if basic else 0,
            "protocol_id":protocol, "max_model_calls":None,"max_tool_calls":None,"max_output_tokens":None,
            "hosted_timeout_minutes":60, "controller_timeout_seconds":3600,
            "final_action":"repair_submit_edits", "harness":"null", "tool_choice":"required",
            "initial_fault": "reversed_controls", "practice_seeds": "2040000..2040031; failure search 2040000+32*r .. 2040031+32*r",
            "promotion_seeds": "2060000+512*r .. 2060511+512*r", "final_audit_seeds": [2090000, 2091023],
            "promotion_rule": "prevent recorded miss AND changed>0 AND gain>0.10 AND gain-2*paired_SE>0",
            "protocol": "Observe a miss, infer a cause, use tools and practice simulations, submit diagnosis and edits. Freeze history and randomness for targeted replay; also require broad improvement. No gradients. Final audit never used for promotion.",
            "model_repo": MODEL_ID, "space_repo": SPACE_ID}
    if basic:
        plan.update(practice_seeds="3040000..3040031; diagnostic failure seed 314159",
                    promotion_seeds="3060000..3060511", final_audit_seeds=[3090000,3091023],
                    promotion_rule="changed>0 AND gain>0.10 AND gain-2*paired_SE>0",
                    target_replay_required=False,
                    protocol="One tool-assisted planted-fault repair. Broad improvement decides promotion; the recorded miss is diagnostic only. No golden edit or solver supplied. Final audit never used for promotion. This is not an unknown-failure or recursive-repair claim.")
    if qualified:
        plan.update(final_action="repair_propose_edits",
                    submission_rule="Automatically freeze the FIRST proposal with changed>0 AND practice gain>0.10 AND practice gain-2*paired_SE>0",
                    practice_seeds="4040000..4040031; diagnostic failure seed 314159",
                    promotion_seeds="4060000..4060511", final_audit_seeds=[4090000,4091023],
                    protocol="One tool-assisted planted-fault repair. The first practice-qualified raw LLM weight proposal ends editing automatically. The independent promotion gate may reject it; a separate final audit never selects edits. No golden edit, numerical candidate solver, or gradient training is supplied.")
    save(RUN / "plan.json", plan)
    base = faulty_policy("reversed_controls")
    save(RUN / "current-policy.json", base.sparse())
    save(RUN / "state.json", {"completed_rounds": 0, "current_hash": fingerprint(base), "history": []})
    return plan


def submit(index):
    plan = initialize()
    state = read(RUN / "state.json")
    folder = RUN / f"round-{index:02d}"
    folder.mkdir(exist_ok=True)
    if (folder / "submission.json").exists():
        return read(folder / "submission.json")
    if (folder / "submit-output.txt").exists():
        output = (folder / "submit-output.txt").read_text(encoding="utf-8")
        matches = re.findall(r"(?:evaluations/|Evaluation ID:\s*|evaluation_id[\"\s:]+)([A-Za-z0-9_-]+)", output)
        if not matches:
            raise RuntimeError("Inspect saved submission output; do not duplicate the hosted call.")
        record = {"evaluation_id": matches[0], "round_index": index, "baseline_hash": state["current_hash"]}
        save(folder / "submission.json", record)
        return record
    if index != state["completed_rounds"] or index >= plan["rounds"]:
        raise ValueError("Rounds must run once, in order")
    if index and not (RUN / f"round-{index-1:02d}" / "deployment-check.json").exists():
        raise RuntimeError("Verify the previous deployed checkpoint before advancing")
    baseline = current_policy(state)
    snapshot = json.dumps(baseline.sparse(), separators=(",", ":"))
    feedback = state["history"][-1].get("feedback", {}) if state["history"] else {}
    env_args = {"iterative": True, "policy_snapshot_b64": base64.b64encode(snapshot.encode()).decode(), "policy_sha256": state["current_hash"],
                "round_index": index, "final_round": index == plan["rounds"]-1,
                "previous_feedback_b64": base64.b64encode(json.dumps(feedback).encode()).decode()}
    if plan.get("protocol_id") in ("failure-driven-v1","failure-driven-v2","basic-repair-v1","qualified-repair-v1"):
        seed = 314159
        if index:
            deployed = read(RUN / f"round-{index-1:02d}" / "deployment-check.json")
            if deployed["policy_hash"] != state["current_hash"]:
                raise ValueError("Failure handoff belongs to different deployed weights")
            seed = deployed["seed"] if deployed["reason"] == "miss" else -1
        env_args.update(failure_driven=True,failure_seed=seed)
        if plan.get("protocol_id") in ("basic-repair-v1","qualified-repair-v1"):
            env_args["basic_repair"] = True
        if plan.get("protocol_id") == "failure-driven-v1":
            env_args["max_editor_turns"] = plan["max_model_calls"]
    save(folder / "environment-args.json", env_args)
    save(folder / "baseline-policy.json", json.loads(snapshot))
    # TOML avoids shell quoting of the nested snapshot and keeps the request reviewable.
    config = f'model = {json.dumps(plan["editor"])}\nnum_examples = 1\nrollouts_per_example = 1\nmax_tokens = {plan["max_tokens"]}\ntemperature = {plan["temperature"]}\nmax_concurrent = 1\n\n[[eval]]\nenv_id = {json.dumps(plan["environment"])}\n[eval.env_args.taskset]\n'
    config += "\n".join(k + " = " + (str(v).lower() if isinstance(v, bool) else json.dumps(v)) for k, v in env_args.items()) + "\n"
    if plan.get("protocol_id") == "failure-driven-v1":
        config += f'\n[eval.env_args.agent]\nmax_turns = {plan["max_model_calls"]}\nmax_output_tokens = {plan["max_output_tokens"]}\n'
    elif plan.get("protocol_id") in ("failure-driven-v2","basic-repair-v1","qualified-repair-v1"):
        config += '\n[eval.env_args.taskset.task]\nprepared_tools = true\n'
        if plan.get("protocol_id") == "qualified-repair-v1":
            config += 'qualified_proposals = true\n'
        # v1 AgentConfig defaults to None: omitting these is genuinely unlimited,
        # rather than a large disguised limit. Runtime service timeouts still apply.
        config += '\n[eval.env_args.agent.harness]\nid = "null"\n'
        config += '\n[eval.env_args.agent.sampling]\ntool_choice = "required"\nparallel_tool_calls = false\n'
    config_path = folder / "eval.toml"
    config_path.write_text(config, encoding="utf-8")
    path = str(config_path)
    if os.name == "nt":
        path = "/mnt/" + path[0].lower() + path[2:].replace("\\", "/")
    output = prime(["eval", "run", path, "--hosted", "--eval-name", f"weight-repair-loop-round-{index+1}", "--timeout-minutes", str(plan.get("hosted_timeout_minutes",60)), "--plain"])
    (folder / "submit-output.txt").write_text(output, encoding="utf-8")
    print(output, flush=True)
    matches = re.findall(r"(?:evaluations/|Evaluation ID:\s*|evaluation_id[\"\s:]+)([A-Za-z0-9_-]+)", output)
    if not matches:
        raise RuntimeError("Inspect saved submission output for the evaluation ID")
    record = {"evaluation_id": matches[0], "round_index": index, "baseline_hash": state["current_hash"]}
    save(folder / "submission.json", record)
    return record


def check_comparison(report, count, start):
    """Check recorded returns and arithmetic, without rerunning any games."""
    if (report["episodes"], report["seed_start"], report["seed_end"]) != (count, start, start+count-1):
        raise ValueError("Unexpected evaluation seed range")
    a, b = np.asarray(report["baseline_returns"], dtype=float), np.asarray(report["candidate_returns"], dtype=float)
    if a.shape != (count,) or b.shape != (count,) or not np.isfinite([a,b]).all():
        raise ValueError("Invalid episode returns")
    if any(np.any((x < -1) | (x > 24) | (x != np.floor(x))) for x in (a,b)):
        raise ValueError("Returns are outside game reward bounds")
    delta = b-a
    expected = {"baseline_return":a.mean(), "candidate_return":b.mean(), "gain":delta.mean(),
                "paired_standard_error":delta.std(ddof=1)/np.sqrt(count)}
    if any(not np.isclose(report[k], v, rtol=1e-10, atol=1e-10) for k,v in expected.items()):
        raise ValueError("Verifier summary disagrees with recorded returns")


def check_first_qualification(trace, receipt, final_tool):
    """Audit ordered proposal receipts; a practice pass must be the terminal edit."""
    pending, seen = {}, set()
    committed = False
    valid_proposals = 0
    for node in trace["nodes"]:
        message = node["message"]
        role = message["role"]
        if committed and role in ("assistant","tool"):
            raise ValueError("Editor continued after the first qualifying proposal")
        if role == "assistant":
            for tool in message.get("tool_calls") or []:
                if tool["name"] not in ("repair_inspect_policy","repair_propose_edits"):
                    raise ValueError("Unexpected tool in qualified editor trace")
                if tool["id"] in seen:
                    raise ValueError("Repeated tool call identity")
                seen.add(tool["id"])
                pending[tool["id"]] = tool
        if role != "tool":
            continue
        tool = pending.pop(message["tool_call_id"],None)
        if tool is None or message.get("name") != tool["name"]:
            raise ValueError("Tool result does not match a preceding request")
        if tool["name"] != "repair_propose_edits":
            continue
        try:
            result = json.loads(message["content"])
        except (ValueError,TypeError):
            continue  # Validation failures can be plain-text tool errors.
        if not isinstance(result,dict) or result.get("valid") is not True:
            continue
        valid_proposals += 1
        qualifies = bool(result["changed_parameters"] > 0 and result["practice_gain"] > 0.10
                         and result["practice_gain"] > 2*result["practice_paired_standard_error"])
        if result.get("practice_qualified") is not qualifies or result.get("submitted") is not qualifies:
            raise ValueError("Practice qualification and submission receipt disagree")
        if result.get("valid_proposal_number") != valid_proposals:
            raise ValueError("Proposal receipt sequence disagrees with trace")
        if qualifies:
            if tool["id"] != final_tool["id"] or result.get("submission_sha256") != receipt["sha256"]:
                raise ValueError("The first qualifying proposal was not the frozen final edit")
            committed = True
    # VF may stop when shared state commits, before the final tool response is
    # appended. The trusted state receipt and independently reproduced practice
    # check still prove that exact final proposal qualified. Earlier results must
    # all be present, nonqualifying, and counted; never infer a receipt from args.
    if not committed and set(pending) == {final_tool["id"]}:
        if receipt.get("valid_proposal_number") != valid_proposals+1:
            raise ValueError("Terminal state receipt disagrees with earlier proposal count")
        return False
    if not committed or pending or receipt.get("valid_proposal_number") != valid_proposals:
        raise ValueError("Missing final qualification receipt or unfinished tool request")
    return True


def validate_trace(trace, summary, submission, plan, baseline, index):
    qualified = plan.get("protocol_id") == "qualified-repair-v1"
    basic = plan.get("protocol_id") in ("basic-repair-v1","qualified-repair-v1")
    failure_driven = plan.get("protocol_id") in ("failure-driven-v1","failure-driven-v2","basic-repair-v1","qualified-repair-v1")
    prepared = plan.get("protocol_id") in ("failure-driven-v2","basic-repair-v1","qualified-repair-v1")
    if trace["run"]["id"] != submission["evaluation_id"] or summary["environment_id"] != plan["environment"]:
        raise ValueError("Trace belongs to a different run or environment")
    if not trace.get("ok") or not trace.get("is_completed") or trace.get("is_timeout") or trace.get("errors"):
        raise ValueError("Incomplete or errored trace; do not silently retry an editor attempt")
    activity = summary["activity"]
    # Hitting the declared turn limit without an answer is a rejected attempt,
    # not a transport failure. Never salvage a patch from pending tool arguments.
    turn_limit = bool(failure_driven and summary["execution"].get("is_truncated")
                      and summary.get("score",{}).get("outcome") == "max_turns"
                      and trace.get("stop_condition") == "max_turns"
                      and activity["model_calls"] == plan["max_model_calls"])
    token_limit = bool(failure_driven and summary["execution"].get("is_truncated")
                       and summary.get("score",{}).get("outcome") == "agent_completed"
                       and trace.get("stop_condition") == "agent_completed"
                       and trace.get("calls") and trace["calls"][-1]["finish_reason"] == "length"
                       and trace["calls"][-1]["usage"]["completion_tokens"] == plan["max_tokens"])
    exhausted = turn_limit or token_limit
    if failure_driven:
        if activity["model_calls"] < 1 or (plan["max_model_calls"] is not None and activity["model_calls"] > plan["max_model_calls"]) or (plan["max_tool_calls"] is not None and activity["tool_calls"] > plan["max_tool_calls"]):
            raise ValueError("Tool-assisted editor exceeded its recorded budget")
    elif activity["model_calls"] != 1 or activity["tool_calls"] != 0 or activity["tool_results"] != 0:
        raise ValueError("Protocol requires exactly one model response and no tool use")
    if summary["execution"]["has_error"] or (summary["execution"]["is_truncated"] and not exhausted):
        raise ValueError("Trace execution failed: " + str(summary.get("score",{}).get("outcome","unknown stop")))
    if len(trace["calls"]) != activity["model_calls"]:
        raise ValueError("Unexpected number of model calls")
    completed_calls = []
    recovered_requests = []
    for request in trace["calls"]:
        if request.get("error"):
            error = request["error"]
            # VF records a rejected HTTP request without an assistant node,
            # finish reason, or token usage. A later successful retry is not
            # another proposal. Never ignore partial output or terminal errors.
            if (not prepared or error.get("type") != "ProviderError"
                    or error.get("status_code") not in (408,429,500,502,503,504)
                    or request.get("node") is not None
                    or request.get("finish_reason") is not None
                    or request.get("usage")):
                raise ValueError("Unrecoverable or partially generated model request")
            recovered_requests.append(request)
        else:
            completed_calls.append(request)
    if not completed_calls or trace["calls"][-1].get("error"):
        raise ValueError("No completed final model response")
    call = completed_calls[-1]
    if (call.get("finish_reason") != ("tool_calls" if prepared else "stop") and not exhausted) or any(c["model"] != plan["editor"] for c in trace["calls"]) or any(c.get("finish_reason") not in ("stop","tool_calls") and not (token_limit and i == len(completed_calls)-1 and c.get("finish_reason") == "length") for i,c in enumerate(completed_calls)):
        raise ValueError("Wrong editor or incomplete generation")
    if any(c["sampling"].get("temperature") != plan["temperature"] or c["sampling"].get("max_tokens") != plan["max_tokens"] for c in trace["calls"]):
        raise ValueError("Sampling settings changed")
    if failure_driven and plan["max_output_tokens"] is not None and sum(c["usage"]["completion_tokens"] for c in completed_calls) > plan["max_output_tokens"]:
        raise ValueError("Editor exceeded total output token budget")
    replies = [n["message"] for n in trace["nodes"] if n["message"]["role"] == "assistant"]
    if not replies or (not failure_driven and (len(replies) != 1 or any(n["message"].get("tool_calls") or n["message"]["role"] == "tool" for n in trace["nodes"]))):
        raise ValueError("Unexpected assistant or tool messages")
    response = replies[-1].get("content") or ""
    report = trace["info"]["repair_pipeline"]
    if prepared:
        if trace.get("stop_condition") != "edit_submitted" or not trace["info"].get("editor_setup",{}).get("ready"):
            raise ValueError("Prepared editor did not finish with a committed edit")
        final_tools = replies[-1].get("tool_calls") or []
        final_action = "repair_propose_edits" if qualified else "repair_submit_edits"
        if plan.get("final_action",final_action) != final_action:
            raise ValueError("Frozen final action disagrees with protocol")
        if len(final_tools) != 1 or final_tools[0].get("name") != final_action:
            raise ValueError("Final action must be exactly one " + final_action + " call")
        response = final_tools[0]["arguments"]
        submission_record = report.get("submission",{})
        if submission_record.get("source") != final_action or json.loads(response) != json.loads(submission_record["json"]) or hashlib.sha256(submission_record["json"].encode()).hexdigest() != submission_record["sha256"]:
            raise ValueError("Final edit differs from the tool's committed submission")
        if qualified:
            check_first_qualification(trace,submission_record,final_tools[0])
        if report.get("protocol") != plan["protocol_id"]:
            raise ValueError("Wrong prepared-editor verifier protocol")
        if basic and report.get("target_replay_required") is not False:
            raise ValueError("Basic repair must declare its diagnostic-only replay")
        if any(c["sampling"].get("tool_choice") != "required" or c["sampling"].get("parallel_tool_calls") is not False for c in trace["calls"]):
            raise ValueError("Required sequential tool-calling settings changed")
    if exhausted and (report["valid"] or report["accepted"] or report.get("candidate_hash") or report.get("edits") or report.get("reward") != 0.):
        raise ValueError("Truncated attempts cannot promote or supply a candidate")
    data = trace["task"]["data"]
    if prepared and data.get("protocol_id",plan["protocol_id"]) != plan["protocol_id"]:
        raise ValueError("Task and recorded protocol disagree")
    if bool(data.get("failure_case_json")) != failure_driven:
        raise ValueError("Wrong repair protocol in hosted task")
    if failure_driven:
        case = json.loads(data["failure_case_json"])
        if case != report["failure_case"] or case_digest(case) != case["case_id"] or case["policy_hash"] != fingerprint(baseline):
            raise ValueError("Failure handoff identity mismatch")
        before = report["baseline_replay"]
        if before["case_id"] != case["case_id"] or before["prevented_recorded_miss"] or before["reason"] != "miss" or not before["same_ball_contact"]:
            raise ValueError("Baseline failed to reproduce the target miss")
    if data["round_index"] != index or fingerprint(from_sparse(data["loop_policy"])) != fingerprint(baseline):
        raise ValueError("Hosted task did not use the submitted policy")
    if report["round_index"] != index or report["baseline_hash"] != submission["baseline_hash"] or fingerprint(baseline) != report["baseline_hash"]:
        raise ValueError("Baseline identity mismatch")
    if report["verifier"] != plan["environment"].split("/",1)[1]:
        raise ValueError("Unexpected verifier version")
    candidate, valid = baseline, False
    try:
        if exhausted:
            raise ValueError("No final submission before the editor budget limit")
        elif failure_driven:
            payload, explanation = parse_failure_submission(response)
        else:
            payload = parse_submission(response)
        candidate = apply_edits(baseline, payload)
        valid = True
    except (ValueError, TypeError, OverflowError, RecursionError):
        pass
    if valid != report["valid"]:
        raise ValueError("Editor response validity disagrees with verifier")
    if valid:
        if payload != report["edits"] or fingerprint(candidate) != report["candidate_hash"]:
            raise ValueError("Candidate identity mismatch")
        changed = sum(int(np.count_nonzero(candidate.weights[k] != baseline.weights[k])) for k in SHAPES)
        if qualified:
            check_comparison(report["practice_qualification"],32,4040000)
            if not passes_gate(report["practice_qualification"],changed):
                raise ValueError("Submitted edit did not pass the frozen practice qualification rule")
        check_comparison(report["gate"],512,(4060000 if qualified else (3060000 if basic else (2060000 if prepared else (1060000 if failure_driven else 60000))))+512*index)
        accepted = passes_gate(report["gate"],changed)
        if failure_driven:
            replay = report["candidate_replay"]
            if explanation != report["explanation"] or replay["case_id"] != case["case_id"]:
                raise ValueError("Diagnosis or replay identity mismatch")
            if replay["prevented_recorded_miss"] and (not replay["same_ball_contact"] or replay["reason"] == "miss" or replay["ball"][0] != 8 or replay["step"] != case["failure_step"]):
                raise ValueError("Claimed prevention disagrees with replay outcome")
            if not basic:
                accepted = bool(accepted and replay["prevented_recorded_miss"])
        if changed != report["changed_parameters"] or accepted != report["accepted"]:
            raise ValueError("Promotion decision disagrees with declared gate")
    elif report["accepted"]:
        raise ValueError("Invalid candidate was accepted")
    selected = candidate if report["accepted"] else baseline
    if fingerprint(selected) != report["selected_hash"] or trace["rewards"]["improvement"]["score"] != float(report["accepted"]):
        raise ValueError("Selected checkpoint or reward mismatch")
    if bool(data["final_round"]) != (index == plan["rounds"]-1):
        raise ValueError("Unexpected final-audit scheduling")
    if index == plan["rounds"]-1:
        check_comparison(report["final_audit"],1024,4090000 if qualified else (3090000 if basic else (2090000 if prepared else (1090000 if failure_driven else 90000))))
    elif "final_audit" in report:
        raise ValueError("Final audit was revealed before the last round")
    editor = {"trace_id":trace["id"], "evaluation_id":submission["evaluation_id"], "model":call["model"],
              "prompt":data["prompt"], "response":response, "call":call, "calls":trace["calls"],
              "activity":activity, "advertised_tools":[t.get("name",t.get("function",{}).get("name")) for t in summary.get("tool_definitions",[])],
              "protocol_note":("Tools and practice simulations permitted under the run's recorded budget; scoring runs outside the editor's writable runtime." if failure_driven else "The hosted harness advertised tools. The prompt prohibited tool use; the trace confirms zero tool calls. Tools were not technically unavailable.")}
    editor["outcome"] = "turn_limit_without_submission" if turn_limit else ("token_limit_without_submission" if token_limit else ("submitted" if valid else "invalid_submission"))
    if prepared:
        editor["setup"] = trace["info"]["editor_setup"]
        editor["submission_receipt"] = report["submission"]
        editor["recovered_provider_requests"] = len(recovered_requests)
        editor["submission_mode"] = "first_practice_qualified_proposal" if qualified else "explicit_final_submission"
    if exhausted:
        # Surface actual tool errors to the next attempt, without exporting raw
        # tool logs or changing the environment, seeds, or budget mid-experiment.
        errors = []
        for node in trace["nodes"]:
            message = node["message"]
            if message["role"] == "tool" and isinstance(message.get("content"),str):
                for line in message["content"].splitlines():
                    if line.startswith(("ModuleNotFoundError:","AttributeError:","ImportError:")):
                        errors.append(line[:300])
        editor["tool_errors"] = list(dict.fromkeys(errors))[-5:]
    return candidate, selected, report, editor


def collect(index):
    plan, state = initialize(), read(RUN / "state.json")
    if index < state["completed_rounds"]:
        return state["history"][index]
    if index != state["completed_rounds"]:
        raise ValueError("Collect rounds in order")
    folder = RUN / f"round-{index:02d}"
    submission = read(folder / "submission.json")
    status = json_output(prime(["eval","get",submission["evaluation_id"],"--output","json","--plain"]))
    save(folder / "hosted-status.json", status)
    if status["status"] != "COMPLETED":
        raise RuntimeError("Hosted evaluation is " + status["status"])
    summaries = json_output(prime(["traces","list","--run-id",submission["evaluation_id"],"--output","json","--plain"]))
    if len(summaries["items"]) != 1 or summaries.get("next_cursor"):
        raise ValueError("Expected exactly one trace")
    summary = summaries["items"][0]
    save(folder / "trace-summary.json", summary)
    if (folder / "trace.json").exists():
        trace = read(folder / "trace.json")
    else:
        trace = json_output(prime(["traces","get",summary["trace_id"],"--raw","--output","json","--plain"]))
        save(folder / "trace.json", trace)
    if trace["id"] != summary["trace_id"]:
        raise ValueError("Trace identity mismatch")
    baseline = from_sparse(json.dumps(read(folder / "baseline-policy.json")))
    if fingerprint(baseline) != state["current_hash"]:
        raise ValueError("Incumbent changed since submission")
    candidate, selected, report, editor = validate_trace(trace,summary,submission,plan,baseline,index)
    save(folder / "candidate-policy.json",candidate.sparse())
    save(folder / "verifier-report.json",report)
    save(folder / "editor-response.json",editor)
    if "failure_case" in report:
        save(folder / "failure-case.json",report["failure_case"])
        save(folder / "failure-replay.json",{"baseline":report["baseline_replay"],"candidate":report.get("candidate_replay")})
    feedback = {"accepted":report["accepted"], "valid":report["valid"],
                "edits":report.get("edits"), "changed_parameters":report.get("changed_parameters",0)}
    if report["valid"]:
        feedback.update({k:report["gate"][k] for k in ("baseline_return","candidate_return","gain","paired_standard_error")})
    else:
        feedback["error"] = report.get("error")
    feedback["outcome"] = editor["outcome"]
    if editor["outcome"] in ("turn_limit_without_submission","token_limit_without_submission"):
        limit = "model-turn" if editor["outcome"] == "turn_limit_without_submission" else "per-response output-token"
        feedback.update(error=f"The previous attempt exhausted its {limit} budget without a final JSON submission; no weight edit was evaluated.",
                        model_calls=editor["activity"]["model_calls"],tool_calls=editor["activity"]["tool_calls"],
                        tool_errors=editor.get("tool_errors",[]),
                        tool_guidance="Reserve the final model turn for diagnosis/expected_prevention/edits JSON. Keep each response concise within 4096 tokens, including any reasoning tokens. Policy arrays are in policy.weights['W1'], not policy.W1. The snapshot file is a JSON list of nonzero assignments. The helper reads repair_lab.zip directly; no unzip is necessary. Install numpy and run python /tmp/repair_practice.py in one tool call if needed. Submit your best bounded proposal before the turn limit even if practice exploration is incomplete.")
    if "failure_case" in report:
        feedback.update(explanation=report.get("explanation"),failure_case_id=report["failure_case"]["case_id"],
                        prevented_recorded_miss=report.get("candidate_replay",{}).get("prevented_recorded_miss",False))
    item = {"round":index, "accepted":report["accepted"], "baseline_hash":report["baseline_hash"],
            "candidate_hash":report.get("candidate_hash"), "selected_hash":report["selected_hash"],
            "evaluation_id":submission["evaluation_id"], "trace_id":trace["id"], "feedback":feedback}
    state["history"].append(item)
    state.update(completed_rounds=index+1,current_hash=report["selected_hash"])
    if "final_audit" in report:
        state["final_audit"] = report["final_audit"]
    # State is authoritative; recover the lightweight snapshot from its saved report on resume.
    save(RUN / "state.json",state)
    save(RUN / "current-policy.json",selected.sparse())
    print(json.dumps(item,indent=2),flush=True)
    return item


def archive_failure(index, status):
    """Preserve a failed service attempt without advancing the repair history."""
    folder = RUN / f"round-{index:02d}"
    save(folder / "hosted-status.json", status)
    summaries = json_output(prime(["traces","list","--run-id",status["evaluation_id"],"--output","json","--plain"]))
    save(folder / "trace-list.json", summaries)
    items = summaries.get("items", [])
    if len(items) > 1 or summaries.get("next_cursor"):
        raise ValueError("Unexpected multiple traces in a single-attempt run")
    if items:
        summary = items[0]
        trace = json_output(prime(["traces","get",summary["trace_id"],"--raw","--output","json","--plain"]))
        save(folder / "trace-summary.json", summary)
        save(folder / "trace.json", trace)
        report = summarize_failure(trace,summary,status,index)
    else:
        report = {"round":index,"evaluation_id":status["evaluation_id"],"trace_id":None,
                  "status":status["status"],"environment":initialize()["environment"],
                  "outcome":"service_failure_without_trace","errors":[{"message":status.get("error_message")}],
                  "activity":{},"valid_practice_results":None,"final_submission_calls":None,
                  "independent_verifier_completed":None,"trace_available":False,"promoted":False,
                  "note":"No trace was available from Prime. Missing performance measurements are not zero scores. No weights promoted; preserve this attempt before any retry."}
    (folder / "hosted-log.txt").write_text(prime(["eval","logs",status["evaluation_id"],"--tail","200","--plain"]),encoding="utf-8")
    save(folder / "execution-failure.json", report)
    state = read(RUN / "state.json")
    failures = state.setdefault("execution_failures", [])
    failures[:] = [f for f in failures if f["evaluation_id"] != status["evaluation_id"]]
    failures.append(report)
    save(RUN / "state.json",state)
    return report


def publish_model(force=False):
    from huggingface_hub import HfApi, hf_hub_download
    from safetensors.numpy import save_file
    plan = initialize()
    state = read(RUN / "state.json")
    if not force and state.get("published_rounds") == state["completed_rounds"]:
        return state["model_revision"]
    api = HfApi()
    if api.whoami()["name"].lower() != MODEL_ID.split("/")[0].lower():
        raise RuntimeError("Unexpected Hugging Face account")
    if not state.get("model_created"):
        if api.repo_exists(MODEL_ID):
            raise RuntimeError("Target repository already exists; refuse to overwrite an unrelated run")
        api.create_repo(MODEL_ID, private=False)
        state["model_created"] = True
        save(RUN / "state.json", state)
    head = api.model_info(MODEL_ID).sha
    if state.get("model_revision") and head != state["model_revision"]:
        raise RuntimeError("Remote model changed outside this controller")
    folder = RUN / "model-export"
    folder.mkdir(exist_ok=True)
    policy = current_policy(state)
    save_file(policy.weights, folder / "model.safetensors", metadata={"format":"numpy"})
    save_file(faulty_policy("reversed_controls").weights, folder / "initial_baseline.safetensors", metadata={"format":"numpy"})
    policy.save(folder / "policy.npz")
    save(folder / "policy.json", policy.sparse())
    last_accepted = next((item["round"]+1 for item in reversed(state["history"]) if item["accepted"]), None)
    manifest = {"model_id": MODEL_ID, "policy_hash": fingerprint(policy),
                "weights_file_sha256": hashlib.sha256((folder / "model.safetensors").read_bytes()).hexdigest(),
                "label": f"Accepted repair from round {last_accepted}" if last_accepted else "Initial faulty reversed-controls policy",
                "history": state["history"], "parent_revision": state.get("model_revision"),
                "final_audit": state.get("final_audit"), "run_plan": plan}
    if state.get("setup_failures"):
        manifest["setup_failures"] = state["setup_failures"]
    if state.get("execution_failures"):
        manifest["execution_failures"] = state["execution_failures"]
    if (RUN / "wandb-sync.json").exists():
        manifest["wandb_project_url"] = read(RUN / "wandb-sync.json")["project_url"]
    save(folder / "manifest.json", manifest)
    save(folder / "config.json", {"architecture":"tanh_mlp","dimensions":[500,32,32,5],"parameters":17253,
                                   "environment":plan["environment"],"policy_hash":fingerprint(policy)})
    shutil.copy2(ROOT / "deployment/inference.py", folder / "inference.py")
    shutil.copy2(ROOT / "LICENSE", folder / "LICENSE")
    for name in ("PREPARED_EDITOR.md","FAILURE_DRIVEN.md","LANGUAGE_EXTENSION.md","BASIC_REPAIR.md","QUALIFIED_REPAIR.md","TRACKING.md"):
        shutil.copy2(ROOT / name, folder / name)
    audit = folder / "experiment"
    audit.mkdir(exist_ok=True)
    shutil.copy2(RUN / "plan.json", audit / "plan.json")
    for failure in state.get("execution_failures", []):
        save(audit / "failed-attempts" / (failure["evaluation_id"]+".json"),failure)
    for index in range(state["completed_rounds"]):
        source = RUN / f"round-{index:02d}"
        target = audit / f"round-{index:02d}"
        target.mkdir(exist_ok=True)
        for name in ("baseline-policy.json", "candidate-policy.json", "verifier-report.json", "editor-response.json", "submission.json", "deployment-check.json", "failure-case.json", "failure-replay.json"):
            if (source / name).exists():
                shutil.copy2(source / name, target / name)
    rows = ["| Round | Incumbent | Candidate | Mean gain | Decision |", "| --- | ---: | ---: | ---: | --- |"]
    for item in state["history"]:
        f = item["feedback"]
        if "gain" in f:
            rows.append(f"| {item['round']+1} | {f['baseline_return']:.4f} | {f['candidate_return']:.4f} | {f['gain']:+.4f} | {'Accepted' if item['accepted'] else 'Rejected'} |")
        else:
            reason = {"turn_limit_without_submission":"No submission: turn limit", "token_limit_without_submission":"No submission: token limit"}.get(f.get("outcome"),"Invalid submission")
            rows.append(f"| {item['round']+1} | — | {reason} | — | Rejected |")
    card = "---\nlicense: mit\ntags:\n- reinforcement-learning\n- breakout\n- weight-space-programming\n- numpy\n- safetensors\n---\n# Iterative Weight Space Breakout\n\n"
    card += f"**Current weights: {manifest['label']}.** This is a 17,253-parameter game controller, not a language model.\n\n"
    card += f"[Live simulation](https://huggingface.co/spaces/{SPACE_ID}) · [Pipeline source](https://github.com/HarleyCoops/weight-space-breakout-pipeline) · [Prime environment](https://app.primeintellect.ai/dashboard/environments/harleycooper/weight-space-breakout)\n\n"
    if manifest.get("wandb_project_url"):
        card += f"[Weights & Biases experiment records]({manifest['wandb_project_url']}) track practice, independent verification, selected checkpoints, and execution failures separately.\n\n"
    for failure in state.get("execution_failures", []):
        if failure.get("trace_available") is False:
            detail = "Prime has not supplied a trace. Model turns and performance measurements are unavailable; missing results are not zero scores."
        else:
            detail = f"It recorded {failure.get('activity',{}).get('model_turns',0)} model turns, {failure['valid_practice_results']} valid practice results, and {failure['final_submission_calls']} final submission calls."
        card += f"**Hosted attempt stopped: {failure['outcome']}.** {detail} No weights were promoted from this failed attempt. Practice results are not held-out evaluation results. [Evaluation](https://app.primeintellect.ai/dashboard/evaluations/{failure['evaluation_id']}) · [Failure audit](experiment/failed-attempts/{failure['evaluation_id']}.json).\n\n"
    if plan.get("compute_platform") == "huggingface-jobs":
        card += "Qwen inference ran through Prime Inference. The existing environment, practice tools, independent verifier, and final audit ran on Hugging Face Jobs CPU using the released 0.4.5 task code. This was not a Prime-hosted evaluation. The live simulation also runs on Hugging Face CPU.\n\n"
        if plan.get("job_url"):
            card += f"[Completed HF Job]({plan['job_url']}) · [Full execution evidence](experiment/hf-job/).\n\n"
    else:
        card += "The editor and all experiment game evaluations run on Prime. The simulation runs on Hugging Face CPU. A lightweight controller serializes and publishes verified weights. This iterative feedback protocol is separate from the earlier one-shot experiment.\n\n"
    card += "\n".join(rows) + "\n\nEach valid candidate is tested on 512 fresh matched seeds, so means in different rows are not directly comparable. Attempts without a valid submission receive no candidate evaluation. Rejected attempts remain in the audit; the root checkpoint always contains the incumbent.\n\n"
    if plan.get("protocol_id") == "qualified-repair-v1":
        card += "The LLM proposes raw weight edits using prepared tools. The FIRST proposal with an actual weight change, mean practice gain >0.10, and gain >2 paired standard errors across 32 practice games is automatically frozen and submitted. This ends editing without a separate final-submission decision. Only a separate 512-game promotion gate can replace the published weights; qualification alone cannot. A final 1,024-game audit is never used to select edits. There is no editor-turn cap; the service timeout is 60 minutes. This is one repair of a planted fault, not evidence of sustained recursive improvement. [Exact protocol](QUALIFIED_REPAIR.md).\n\n"
    elif plan.get("protocol_id") == "basic-repair-v1":
        card += "This is one tool-assisted repair of a deliberately damaged teaching policy. The editor can inspect and test proposals without a turn limit, then must explicitly submit its edit. Acceptance requires broad improvement on 512 separate games. The recorded-miss replay is diagnostic only; it is NOT a promotion requirement. The final 1,024-game audit is separate. This trial does not demonstrate unknown-failure repair or sustained recursive improvement. Temperature is 0.6; each response allows 16,384 tokens and the hosted service timeout is 60 minutes.\n\n"
    elif plan.get("protocol_id") == "failure-driven-v2":
        card += "This run uses prepared inspection and practice tools with no model-turn, tool-call, or total-output-token limit. Its final action is the explicit submit_edits tool. Each response has a 16,384-token allowance and hosted jobs retain a 60-minute service timeout. Acceptance requires both catching the recorded miss and broader improvement.\n\n"
    else:
        card += ("This failure-driven run allows tools and up to eight editor calls per round. Acceptance also requires catching the specific recorded miss under a controlled replay.\n\n" if plan.get("protocol_id") == "failure-driven-v1" else "This initial run used one editor response and no tool calls per round.\n\n")
    if state.get("final_audit"):
        a = state["final_audit"]
        card += f"## Final audit\n\nOn {a['episodes']} separate seeds, initial faulty return **{a['baseline_return']:.4f}** and final selected return **{a['candidate_return']:.4f}**; paired gain **{a['gain']:+.4f}**, paired standard error **{a['paired_standard_error']:.4f}**. This final audit was not fed to the editor or used to select a policy.\n\n"
        if plan.get("compute_platform") == "huggingface-jobs":
            differences = np.asarray(a['candidate_returns'])-np.asarray(a['baseline_returns'])
            card += f"The paired games include **{int(np.sum(differences>0))} improvements, {int(np.sum(differences<0))} regressions, and {int(np.sum(differences==0))} ties**. Acceptance means improved average return under the frozen gate; it does not guarantee improvement on every game.\n\n"
            if last_accepted is not None:
                card += f"The accepted candidate changed **{state['history'][last_accepted-1]['feedback']['changed_parameters']} of 17,253 parameters**. It was proposed by Qwen through the practice tools and validated against its exact tool-call arguments before export.\n\n"
        if last_accepted is None:
            card += "No edit was promoted in this bounded run. The published weights remain the initial faulty policy; this run does not demonstrate a policy improvement.\n\n"
        if fingerprint(policy) == fingerprint(tracking_policy()):
            card += "The selected policy exactly matches the intact handwritten reference controller. This demonstrates restoration of the teaching policy; it does not establish sustained recursive improvement or a new learned skill.\n\n"
    card += "## Files\n\n`model.safetensors`, `policy.npz`, and `policy.json` contain the selected policy. `initial_baseline.safetensors` preserves the known-bad start. `manifest.json` links its fingerprint, file checksum, plan, and round decisions. `experiment/` preserves candidates and verifier reports. Every Hub commit is an immutable version; the demo resolves a revision before downloading and checks both hashes.\n\n"
    if plan.get("compute_platform") == "huggingface-jobs":
        card += "## Attribution and scope\n\nThis is an independent replication attempt inspired by [Rikiya Takehi's weight-space programming experiment](https://rikiyat.github.io/blog/weight-space-programming/), its [original logs](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp1_which_game), and [announcement](https://x.com/rikiyatakehi/status/2117313742891327994). Our grid game, handwritten teaching policy, and tool-assisted repair protocol differ from the original experiment. The editor LLM's own weights are unchanged.\n"
    else:
        card += (ROOT / "PIPELINE.md").read_text(encoding="utf-8")
    (folder / "README.md").write_text(card, encoding="utf-8")
    commit = api.upload_folder(repo_id=MODEL_ID, folder_path=folder, parent_commit=head,
                               commit_message=f"Publish verified pipeline state after {state['completed_rounds']} rounds",
                               ignore_patterns=["__pycache__/*", "*.pyc"])
    downloaded = hf_hub_download(MODEL_ID, "model.safetensors", revision=commit.oid)
    if hashlib.sha256(Path(downloaded).read_bytes()).hexdigest() != manifest["weights_file_sha256"]:
        raise RuntimeError("Published checkpoint failed checksum verification")
    state.update(model_revision=commit.oid, published_rounds=state["completed_rounds"])
    save(RUN / "state.json", state)
    if state["history"] and not force:
        save(RUN / f"round-{state['completed_rounds']-1:02d}" / "publication.json", {"revision":commit.oid,"policy_hash":state["current_hash"]})
    print(f"Published {MODEL_ID}@{commit.oid}: {manifest['label']}", flush=True)
    return commit.oid


def verify_deployment(index):
    from gradio_client import Client
    folder = RUN / f"round-{index:02d}"
    publication = read(folder / "publication.json")
    if (folder / "deployment-check.json").exists():
        saved = read(folder / "deployment-check.json")
        if saved["revision"] != publication["revision"]:
            raise ValueError("Deployment record belongs to another revision")
        return saved
    report = read(folder / "verifier-report.json")
    expected = report["selected_episode"]
    result = Client(SPACE_ID,verbose=False).predict(seed=expected["seed"],revision=publication["revision"],api_name="/episode")
    deployed = result[1]
    if deployed["revision"] != publication["revision"] or deployed["policy_hash"] != report["selected_hash"] or deployed["model_id"] != MODEL_ID:
        raise ValueError("Space served a different checkpoint")
    if any(deployed[k] != expected[k] for k in ("seed","step","return","bricks_remaining","reason")):
        raise ValueError("Space episode differs from hosted verifier")
    if deployed["actions"] != [f["action"] for f in expected["frames"][1:]]:
        raise ValueError("Space actions differ from hosted verifier")
    deployed.update(verified=True,checked_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                    comparison="Exact seed, actions, return, termination, and policy hash match the hosted verifier")
    save(folder / "deployment-check.json",deployed)
    print(f"Verified HF simulation round {index+1}: {publication['revision']}",flush=True)
    return deployed


def run_loop():
    plan = initialize()
    if not read(RUN / "state.json").get("space_created"):
        publish_model()
        publish_space()
    for index in range(plan["rounds"]):
        submission = submit(index)
        if index >= read(RUN / "state.json")["completed_rounds"]:
            deadline = time.monotonic()+plan.get("controller_timeout_seconds",900)
            while True:
                status = json_output(prime(["eval","get",submission["evaluation_id"],"--output","json","--plain"]))
                print(f"Round {index+1}: {status['status']}",flush=True)
                if status["status"] == "COMPLETED":
                    break
                if status["status"] in ("FAILED","CANCELLED","STOPPED","ERROR"):
                    archive_failure(index,status)
                    publish_model(force=True)
                    raise RuntimeError("Hosted evaluation stopped; inspect saved ID before any retry")
                if time.monotonic() > deadline:
                    prime(["eval","stop",submission["evaluation_id"],"--plain"])
                    raise TimeoutError("Stopped evaluation after its recorded service timeout")
                time.sleep(15)
            collect(index)
        if index == read(RUN / "state.json")["completed_rounds"]-1:
            publish_model()
        verify_deployment(index)
    publish_model(force=True)


def publish_space():
    from huggingface_hub import HfApi
    state = read(RUN / "state.json")
    if not state.get("model_revision"):
        raise RuntimeError("Publish baseline model first")
    api = HfApi()
    if not state.get("space_created"):
        if api.repo_exists(SPACE_ID, repo_type="space"):
            raise RuntimeError("Space already exists outside this run")
        api.create_repo(SPACE_ID, repo_type="space", space_sdk="gradio", space_hardware="cpu-basic", private=False)
        state["space_created"] = True
        save(RUN / "state.json", state)
    folder = RUN / "space-export"
    folder.mkdir(exist_ok=True)
    for source, target in [("pipeline/space_app.py","app.py"),("weight_space_breakout/game.py","game.py"),("deployment/inference.py","inference.py"),("LICENSE","LICENSE")]:
        shutil.copy2(ROOT / source, folder / target)
    for name in ("PREPARED_EDITOR.md","FAILURE_DRIVEN.md","LANGUAGE_EXTENSION.md","BASIC_REPAIR.md","QUALIFIED_REPAIR.md","TRACKING.md"):
        shutil.copy2(ROOT / name, folder / name)
    app = folder / "app.py"
    app.write_text(app.read_text(encoding="utf-8").replace('MODEL_ID = "HarleyCooper/weight-space-breakout-iterative"', 'MODEL_ID = '+json.dumps(MODEL_ID)), encoding="utf-8")
    (folder / "requirements.txt").write_text("numpy==2.4.1\nsafetensors==0.8.0\nhuggingface_hub==1.27.0\npillow==11.3.0\n")
    (folder / "README.md").write_text(f"---\ntitle: Breakout Weight Repair Loop\nemoji: 🔁\ncolorFrom: blue\ncolorTo: green\nsdk: gradio\nsdk_version: 6.31.0\npython_version: '3.12'\napp_file: app.py\nlicense: mit\nmodels:\n- {MODEL_ID}\n---\n# Hosted repair loop simulation\n\nRuns the currently published, verifier-gated policy. Playing a game does not launch editing.\n\n" + (ROOT / "PIPELINE.md").read_text(encoding="utf-8"), encoding="utf-8")
    head = api.space_info(SPACE_ID).sha
    commit = api.upload_folder(repo_id=SPACE_ID, repo_type="space", folder_path=folder, parent_commit=head,
                               commit_message="Deploy versioned policy simulation and repair-round results")
    state["space_revision"] = commit.oid
    save(RUN / "state.json", state)
    print(f"Published Space https://huggingface.co/spaces/{SPACE_ID}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init", "submit", "collect", "publish-model", "publish-space", "verify-deployment", "run", "sync-wandb"])
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Refresh model documentation/audit without changing weights")
    args = parser.parse_args()
    if args.command == "init":
        print(json.dumps(initialize(), indent=2))
    elif args.command == "submit":
        print(json.dumps(submit(args.round), indent=2))
    elif args.command == "publish-model":
        publish_model(force=args.force)
    elif args.command == "publish-space":
        publish_space()
    elif args.command == "collect":
        collect(args.round)
    elif args.command == "verify-deployment":
        verify_deployment(args.round)
    elif args.command == "run":
        try:
            run_loop()
        finally:
            from pipeline.tracking import maybe_sync
            maybe_sync(RUN)
    elif args.command == "sync-wandb":
        from pipeline.tracking import sync
        print(json.dumps(sync(RUN),indent=2))


if __name__ == "__main__":
    main()
