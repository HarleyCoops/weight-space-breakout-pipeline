"""Check fresh hosted games and exact replays without changing the policy."""
import json
from pathlib import Path
from gradio_client import Client

ROOT = Path(__file__).resolve().parents[1]
client = Client("HarleyCooper/weight-space-breakout-inference", verbose=False, download_files=False)
report = {"fresh_games": [], "replay_matches": []}
for _ in range(3):
    _, game, seed = client.predict("LLM-repaired", api_name="/random_episode")
    assert type(seed) is int and 0 <= seed < 2**32
    assert game["seed"] == seed
    _, replay = client.predict(seed, "LLM-repaired", api_name="/episode")
    assert replay == game
    report["fresh_games"].append(game)
    report["replay_matches"].append(True)
    print(f"Fresh seed {seed}: {game['step']} steps, return {game['return']}; replay matched", flush=True)
assert len({r["seed"] for r in report["fresh_games"]}) == 3
_, fixed_a = client.predict(0, "LLM-repaired", api_name="/episode")
_, fixed_b = client.predict(0, "LLM-repaired", api_name="/episode")
assert fixed_a == fixed_b
assert fixed_a["actions"] == [3, 4, 4, 3, 4, 3]
_, other = client.predict(1, "LLM-repaired", api_name="/episode")
assert fixed_a["actions"] != other["actions"]
report.update(status="passed", same_seed_identical=True, known_seed_behavior_preserved=True, different_seed_actions=True)
(ROOT / "artifacts/hosted-randomization-verification.json").write_text(json.dumps(report, indent=2))
print("Fresh games, seed reporting, exact replays, and original seeded behavior passed.")
