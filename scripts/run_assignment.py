"""Run a named config for seeds in parallel, retaining logs and exit status."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT.parent / "t-zero-learning" / ".env")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               WANDB_PROJECT="sac-assignment", PYTHONUNBUFFERED="1",
               PYTHONIOENCODING="utf-8")
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    jobs = []
    for seed in args.seeds:
        name = f"{args.config}-s{seed}"
        log = (output / f"{name}.log").open("w", encoding="utf-8")
        cmd = [sys.executable, "train.py", "--config", args.config,
               "--override", f"seed={seed}", *args.override]
        started = time.time()
        process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        jobs.append((name, process, log, started, cmd))
        print(f"Started {name}: PID {process.pid}", flush=True)
    failed = False
    for name, process, log, started, cmd in jobs:
        code = process.wait()
        log.close()
        result = {"name": name, "exit_code": code, "started": started,
                  "finished": time.time(), "command": cmd}
        (output / f"{name}.status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Finished {name}: exit {code}", flush=True)
        failed |= code != 0
    raise SystemExit(int(failed))


if __name__ == "__main__":
    main()
