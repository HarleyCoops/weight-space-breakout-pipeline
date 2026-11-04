"""Summarize failed hosted attempts without selecting or promoting practice edits."""
from collections import Counter
import json


def summarize_failure(trace, summary, status, index):
    tools, errors, patches = Counter(), Counter(), Counter()
    practice = []
    qualified_submissions = 0
    for node in trace.get("nodes", []):
        message = node["message"]
        for call in message.get("tool_calls", []):
            name = call["name"]
            tools[name] += 1
            if name in ("repair_test_edits", "repair_propose_edits"):
                try:
                    patch = json.loads(call["arguments"])
                    patches[json.dumps(patch.get("edits",patch), sort_keys=True)] += 1
                except (ValueError, TypeError):
                    pass
        if message.get("role") != "tool" or message.get("name") not in ("repair_test_edits", "repair_propose_edits"):
            continue
        try:
            result = json.loads(message["content"])
        except (ValueError, TypeError):
            errors[message.get("content", "")[:500]] += 1
            continue
        if result.get("valid", True) and "candidate_mean" in result:
            practice.append(result)
            if message.get("name") == "repair_propose_edits" and result.get("submitted") is True:
                qualified_submissions += 1
        else:
            errors[str(result.get("error", "Invalid practice request"))[:500]] += 1
    receipt = trace.get("info",{}).get("repair_pipeline",{}).get("submission",{})
    if receipt.get("source") == "repair_propose_edits" and receipt.get("json") and receipt.get("sha256"):
        # VF may stop on the state commit before appending the terminal result.
        qualified_submissions = max(qualified_submissions,1)
    return {"round": index, "evaluation_id": status["evaluation_id"],
            "trace_id": trace.get("id"), "status": status["status"],
            "environment": summary.get("environment_id"),
            "outcome": trace.get("stop_condition", "execution_failure"),
            "errors": trace.get("errors", []), "activity": summary.get("activity", {}),
            "tool_counts": dict(tools), "valid_practice_results": len(practice),
            "practice_input_errors": dict(errors),
            "target_preventions_in_practice": sum(bool(p.get("target_replay", {}).get("prevented_recorded_miss")) for p in practice),
            "best_practice_mean": max((p["candidate_mean"] for p in practice), default=None),
            "unique_practice_requests": len(patches),
            "largest_identical_request_count": max(patches.values(), default=0),
            "final_submission_calls": tools["repair_submit_edits"] + qualified_submissions,
            "automatically_qualified_submissions": qualified_submissions,
            "independent_verifier_completed": "repair_pipeline" in trace.get("info", {}),
            "promoted": False,
            "note": "Execution failure: no promotion. Practice results are diagnostic only and are not held-out evaluation results."}
