"""Read finished runs from W&B's own append-only local event files."""
import json
from pathlib import Path

from wandb.proto.wandb_internal_pb2 import Record
from wandb.sdk.internal.datastore import DataStore

ROOT = Path(__file__).resolve().parents[1]


def items_to_dict(items):
    result = {}
    for item in items:
        key = item.key or ".".join(item.nested_key)
        result[key] = json.loads(item.value_json)
    return result


def main():
    destination = ROOT / "results" / "histories"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = []
    for path in sorted((ROOT / "wandb").glob("run-*/*.wandb")):
        datastore = DataStore()
        datastore.open_for_scan(str(path))
        run = None
        history = []
        summary = {}
        completed = False
        try:
            while True:
                data = datastore.scan_data()
                if data is None:
                    break
                record = Record()
                record.ParseFromString(data)
                kind = record.WhichOneof("record_type")
                if kind == "run":
                    run = {
                        "id": record.run.run_id, "name": record.run.display_name,
                        "config": items_to_dict(record.run.config.update),
                        "url": f"https://wandb.ai/{record.run.entity}/{record.run.project}/runs/{record.run.run_id}",
                        "source": "local_wandb_event_file",
                    }
                elif kind == "history":
                    history.append(items_to_dict(record.history.item))
                elif kind == "summary":
                    summary.update(items_to_dict(record.summary.update))
                elif kind == "exit":
                    completed = record.exit.exit_code == 0
        except AssertionError:
            # A writer may still be appending to the final block. Never treat
            # partial data as a completed run.
            completed = False
        finally:
            datastore.close()
        if not run or not completed or "eval/mean_return" not in summary:
            continue
        run.update(state="finished", summary=summary, history=history)
        out = destination / f"{run['id']}.json"
        out.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest.append({k: v for k, v in run.items() if k != "history"})
        print(run["id"], run["config"]["exp_name"], len(history), flush=True)
    (ROOT / "results" / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
