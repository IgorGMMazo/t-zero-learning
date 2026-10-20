"""Check completed experiment coverage and submitted configuration provenance."""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    expected = {
        "sac_pendulum": {1},
        "sac_hopper": {1, 2, 3},
        "sac_hopper_early": {1, 2, 3},
        "sac_hopper_slow_q": {1, 2, 3},
        "sac_hopper_utd2": {1, 2, 3},
        "sac_hopper_fixed_alpha": {1, 2},
        "sac_hopper_single_q": {1, 2},
        "ppo_pendulum": {1, 2},
    }
    seen = {name: set() for name in expected}
    checked = []
    for path in (ROOT / "results" / "histories").glob("*.json"):
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["state"] != "finished":
            continue
        config = run["config"]
        name, seed = config["exp_name"], config["seed"]
        if name not in expected:
            continue
        assert seed not in seen[name], f"Duplicate seed: {name} {seed}"
        seen[name].add(seed)
        assert "eval/mean_return" in run["summary"], run["id"]
        source = yaml.safe_load((ROOT / "configs" / f"{name}.yml").read_text(encoding="utf-8"))
        if name.startswith("sac_"):
            for key, value in source["sac"].items():
                assert config["sac"][key] == value, (run["id"], key)
            assert config["total_timesteps"] == source["total_timesteps"], run["id"]
            assert config["env_id"] == source["env_id"], run["id"]
        else:
            assert config["env_id"] == "Hopper-v5", run["id"]
            assert config["total_timesteps"] == 100000, run["id"]
            assert config["ppo_continuous_action"]["gamma"] == .99, run["id"]
            assert run["summary"]["global_step"] == 98304, run["id"]
        checked.append(run["id"])
    assert seen == expected, f"Incomplete coverage: {seen}"
    public = json.loads((ROOT / "results" / "public_access.json").read_text(encoding="utf-8"))
    print(f"Validated {len(checked)} completed runs; configurations and seed coverage match.")
    print("Public-access evidence:", public)


if __name__ == "__main__":
    main()
