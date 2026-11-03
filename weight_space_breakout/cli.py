import argparse
import json
from pathlib import Path
from .evaluation import evaluate, rollout, score_submission
from .policy import FAULTS, Policy, faulty_policy, tracking_policy


def main():
    parser = argparse.ArgumentParser(description="Explore and evaluate weight-space Breakout")
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--policy", choices=["random", "tracker", *FAULTS], default="tracker")
    e.add_argument("--checkpoint", type=Path)
    e.add_argument("--episodes", type=int, default=128)
    e.add_argument("--mode", choices=["sample", "greedy"], default="sample")
    e.add_argument("--output", type=Path)
    d = sub.add_parser("demo")
    d.add_argument("--output", type=Path, default=Path("artifacts/demo.html"))
    s = sub.add_parser("score")
    s.add_argument("submission", type=Path)
    s.add_argument("--fault", choices=FAULTS, default=FAULTS[0])
    args = parser.parse_args()
    if args.command == "demo":
        from .demo import build_demo
        build_demo(args.output)
        print(args.output.resolve())
        return
    if args.command == "score":
        result = score_submission(args.submission.read_text(), args.fault)
    else:
        if not 1 <= args.episodes <= 10000:
            parser.error("episodes must be in [1,10000]")
        policy = Policy.load(args.checkpoint) if args.checkpoint else (
            None if args.policy == "random" else tracking_policy() if args.policy == "tracker" else faulty_policy(args.policy))
        result = evaluate(policy, range(10000, 10000+args.episodes), args.mode)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ("seeds", "returns")}, indent=2))


if __name__ == "__main__":
    main()
