"""Print seed-level diagnostics used to interpret the report figures."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["losses/alpha", "losses/entropy", "charts/q_minus_return",
        "losses/qf1_values", "charts/episodic_return_mean_last100", "charts/SPS"]


def main():
    result = []
    for path in (ROOT / "results" / "histories").glob("*.json"):
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["state"] != "finished":
            continue
        history = run["history"]
        row = {"config": run["config"]["exp_name"], "seed": run["config"]["seed"],
               "id": run["id"], "eval": run["summary"].get("eval/mean_return"),
               "sampled_eval": run["summary"].get("eval/mean_return_sampled"),
               "updates": run["summary"].get("charts/num_updates"),
               "last_global_step": run["summary"].get("global_step"),
               "buffer_size": run["summary"].get("charts/buffer_size")}
        if row["buffer_size"]:
            row["mean_q_minibatch_uses"] = (
                row["updates"] * run["config"]["sac"]["batch_size"] / row["buffer_size"])
        training = [r for r in history if "charts/SPS" in r]
        if training:
            row["training_runtime_minutes"] = training[-1]["_runtime"] / 60
            row["last_return"] = training[-1].get("charts/episodic_return_mean_last100")
        for key in KEYS:
            values = [r[key] for r in history if r.get("global_step", 0) >= 80000 and key in r]
            if values:
                row[key] = {"late_median": statistics.median(values),
                            "late_mean": statistics.mean(values), "min": min(values), "max": max(values)}
        result.append(row)
    result.sort(key=lambda r: (r["config"], r["seed"]))
    destination = ROOT / "results" / "diagnostics.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
