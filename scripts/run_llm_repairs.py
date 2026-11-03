"""Three independent no-tool model calls, followed by frozen-policy evaluation.

Run on Windows with the authenticated Prime CLI in WSL. API secrets are never
read by this script; the CLI uses its existing credential store.
"""
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weight_space_breakout.policy import FAULTS, apply_edits, faulty_policy, parse_submission
from weight_space_breakout.evaluation import TEST_SEEDS, score_submission

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "llm-repair-run"
MODEL = "qwen/qwen3.6-35b-a3b"
HOLDOUT = tuple(range(20000, 21024))


def generate(fault):
    saved = OUT / f"{fault}-response.json"
    if saved.exists():
        return fault, json.loads(saved.read_text())["choices"][0]["message"]["content"]
    prompt = (OUT / f"{fault}-prompt.txt").read_text()
    request = {"model": MODEL, "temperature": 0, "max_tokens": 4096,
               "messages": [{"role": "user", "content": prompt}], "tools": [],
               "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    (OUT / f"{fault}-request.json").write_text(json.dumps(request, indent=2))
    result = subprocess.run(
        ["wsl", "-d", "Ubuntu", "--", "/home/chris/.local/bin/prime", "inference", "chat", MODEL,
         "--temperature", "0", "--max-tokens", "4096", "--output", "json", "--plain"],
        input=prompt.encode(), capture_output=True, timeout=240,
    )
    if result.returncode:
        (OUT / f"{fault}-transport-error.txt").write_bytes(result.stdout + result.stderr)
        raise RuntimeError(f"Prime inference failed for {fault}: exit {result.returncode}")
    raw = result.stdout.decode("utf-8")
    # Prime CLI may prefix JSON output with its human-readable waiting line.
    response = json.loads(raw[raw.index("{"):])
    (OUT / f"{fault}-response.json").write_text(json.dumps(response, indent=2))
    choice = response["choices"][0]
    if choice.get("finish_reason") != "stop" or choice["message"].get("tool_calls"):
        raise RuntimeError(f"Incomplete or tool-calling response for {fault}")
    answer = choice["message"]["content"]
    (OUT / f"{fault}-answer.txt").write_text(answer)
    print(json.dumps({"fault": fault, "generated": True, "usage": response.get("usage")}), flush=True)
    return fault, answer


def main():
    OUT.mkdir(exist_ok=True)
    plan_path = OUT / "experiment-plan.json"
    if plan_path.exists() and "--resume" not in sys.argv:
        raise RuntimeError("Run already started. Preserve it; do not silently rerun model calls.")
    plan = {"model": MODEL, "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "protocol": "One fresh API call per fault; prompt only; no tools, feedback, reference edit, or environment access.",
            "temperature": 0, "max_tokens": 4096, "attempts_per_task": 1,
            "faults": list(FAULTS), "export_task_chosen_before_generation": "reversed_controls",
            "published_test_seeds": list(TEST_SEEDS), "fresh_holdout_seeds": list(HOLDOUT),
            "export_gate": "valid edits and positive mean gain on published seeds and fresh holdout",
            "baseline": "Handwritten faulty teaching policy, not paper PPO checkpoint"}
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan, indent=2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        answers = dict(pool.map(generate, FAULTS))
    # Freeze all submitted policies before any evaluation feedback is produced.
    frozen = {}
    for fault, answer in answers.items():
        try:
            policy = apply_edits(faulty_policy(fault), parse_submission(answer))
            path = OUT / f"{fault}-llm-policy.npz"
            policy.save(path)
            frozen[fault] = hashlib.sha256(path.read_bytes()).hexdigest()
        except (ValueError, TypeError, OverflowError):
            frozen[fault] = None
    (OUT / "frozen-policy-hashes.json").write_text(json.dumps(frozen, indent=2))
    results = {}
    for fault, answer in answers.items():
        results[fault] = {"published_test": score_submission(answer, fault, TEST_SEEDS),
                          "fresh_holdout": score_submission(answer, fault, HOLDOUT),
                          "policy_sha256": frozen[fault]}
        print(json.dumps({"fault": fault, **results[fault]}), flush=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
