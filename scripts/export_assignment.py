"""Download full, unsampled W&B histories for reproducible analysis."""
import json
from pathlib import Path
from dotenv import load_dotenv
import wandb

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "igor_mazo-federal-university-of-goi-s/sac-assignment"


def main():
    load_dotenv(ROOT.parent / "t-zero-learning" / ".env")
    api = wandb.Api(timeout=60)
    destination = ROOT / "results" / "histories"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = []
    for run in api.runs(PROJECT):
        record = {"id": run.id, "name": run.name, "state": run.state,
                  "url": run.url, "config": run.config, "summary": dict(run.summary)}
        path = destination / f"{run.id}.json"
        if path.exists() and run.state == "finished":
            old = json.loads(path.read_text(encoding="utf-8"))
            if old["state"] == "finished":
                manifest.append(record)
                continue
        record["history"] = list(run.scan_history(page_size=1000))
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest.append({k: v for k, v in record.items() if k != "history"})
        print(run.id, run.state, len(record["history"]), flush=True)
    (ROOT / "results" / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
