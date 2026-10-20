"""Aggregate completed seeds without replacing missing measurements with zeros."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["eval/mean_return", "charts/episodic_return_mean_last100"]


def metric(run, key):
    if key in run["summary"]:
        return run["summary"][key]
    points = [row[key] for row in run["history"] if key in row]
    return points[-1] if points else None


def main():
    groups = {}
    for path in (ROOT / "results" / "histories").glob("*.json"):
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["state"] != "finished" or metric(run, "eval/mean_return") is None:
            continue
        name = run["config"]["exp_name"]
        seed = run["config"]["seed"]
        group = groups.setdefault(name, {})
        if seed in group:
            raise ValueError(f"Duplicate completed run for {name} seed {seed}; select explicitly")
        group[seed] = run
    result = {}
    for name, seeds in sorted(groups.items()):
        ordered_runs = [seeds[seed] for seed in sorted(seeds)]
        result[name] = {"seeds": sorted(seeds), "runs": [r["id"] for r in ordered_runs]}
        for key in METRICS:
            values = [metric(r, key) for r in ordered_runs]
            if any(v is None for v in values):
                raise ValueError(f"Missing {key}: {name}")
            result[name][key] = {"mean": statistics.mean(values), "min": min(values),
                                 "max": max(values), "values": values}
    path = ROOT / "results" / "aggregates.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
