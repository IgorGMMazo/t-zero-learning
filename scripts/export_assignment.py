"""Export full W&B histories locally, or download them with --remote."""
import json
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv
import wandb

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "igor_mazo-federal-university-of-goi-s/sac-assignment"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finished-only", action="store_true")
    parser.add_argument("--remote", action="store_true")
    args = parser.parse_args()
    if not args.remote:
        from export_local_history import main as export_local
        export_local()
        return
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT.parent / "t-zero-learning" / ".env")
    api = wandb.Api(timeout=20)
    destination = ROOT / "results" / "histories"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = []
    for run in api.runs(PROJECT):
        if args.finished_only and run.state != "finished":
            continue
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
    for attempt in range(3):
        try:
            main()
            break
        except wandb.errors.CommError:
            if attempt == 2:
                raise
            print(f"Temporary W&B connection failure; retry {attempt + 1}/2.", flush=True)
            time.sleep(2)
