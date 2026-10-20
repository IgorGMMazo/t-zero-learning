"""Continue Part 4 only after all Part 3 seeds finish successfully."""
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CANDIDATES = ["sac_hopper_early", "sac_hopper_slow_q", "sac_hopper_utd2"]
METRIC = "eval/mean_return"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_script(name):
    subprocess.run([sys.executable, f"scripts/{name}.py"], cwd=ROOT, check=True)


def main():
    status_paths = [RESULTS / f"{name}-s{seed}.status.json"
                    for name in ["sac_hopper", *CANDIDATES] for seed in [1, 2, 3]]
    print("Waiting for all twelve Part 3 runs.", flush=True)
    while not all(p.exists() for p in status_paths):
        for path in status_paths:
            if path.exists() and read(path)["exit_code"]:
                raise RuntimeError(f"Training failed: {path}")
        time.sleep(10)
    if any(read(p)["exit_code"] for p in status_paths):
        raise RuntimeError("A Part 3 run failed.")
    run_script("export_assignment")
    run_script("analyze_assignment")
    aggregates = read(RESULTS / "aggregates.json")
    baseline = aggregates["sac_hopper"][METRIC]
    verdicts = {}
    for name in CANDIDATES:
        if len(aggregates[name]["seeds"]) != 3:
            raise ValueError(f"Missing seeds: {name}")
        candidate = aggregates[name][METRIC]
        verdicts[name] = (
            "melhora" if candidate["min"] > baseline["max"] else
            "piora" if candidate["max"] < baseline["min"] else "indistinguível do ruído"
        )
    improvements = [name for name in CANDIDATES if verdicts[name] == "melhora"]
    final = max(improvements, key=lambda n: aggregates[n][METRIC]["mean"]) if improvements else "sac_hopper"
    comparison = final if final != "sac_hopper" else max(CANDIDATES, key=lambda n: aggregates[n][METRIC]["mean"])
    final_runs = [read(p) for p in (RESULTS / "histories").glob("*.json")]
    final_runs = [r for r in final_runs if r["config"]["exp_name"] == final and r["state"] == "finished"]
    alpha_seeds = []
    for run in final_runs:
        values = [r["losses/alpha"] for r in run["history"]
                  if r.get("global_step", 0) >= 80000 and "losses/alpha" in r]
        if not values:
            raise ValueError("No late alpha history")
        alpha_seeds.append({"run": run["id"], "seed": run["config"]["seed"],
                            "median": statistics.median(values), "min": min(values), "max": max(values)})
    alpha = float(f"{statistics.median(r['median'] for r in alpha_seeds):.3g}")
    selection = {"final_config": final, "comparison_config": comparison,
                 "verdicts": verdicts, "fixed_alpha": alpha, "late_alpha_by_seed": alpha_seeds,
                 "selected_at": datetime.now(timezone.utc).isoformat()}
    write(RESULTS / "selection.json", selection)
    base = yaml.safe_load((ROOT / "configs" / f"{final}.yml").read_text(encoding="utf-8"))
    for name, changes in [
        ("sac_hopper_fixed_alpha", {"autotune": False, "alpha": alpha}),
        ("sac_hopper_single_q", {"clipped_double_q": False}),
    ]:
        config = copy.deepcopy(base)
        config["sac"].update(changes)
        (ROOT / "configs" / f"{name}.yml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    predictions = read(RESULTS / "predictions.json")
    predictions["part4_recorded_at"] = datetime.now(timezone.utc).isoformat()
    predictions["part4_reference"] = final
    predictions["alpha_choice"] = (
        f"Alfa fixo {alpha}: mediana das medianas de losses/alpha dos três seeds "
        "entre 80.000 e 100.000 passos, arredondada a três algarismos significativos."
    )
    predictions["part4_alpha"] = (
        f"Com alfa fixo {alpha}, escolhido no regime final da referência, espero menos "
        "regularização inicial do que no autotune, que começa em 1. Isso pode acelerar "
        "a adoção de ações já valorizadas, mas reduzir a diversidade e aumentar "
        "a variação entre seeds. A entropia pode se afastar do alvo -3 porque não há "
        "realimentação ajustando a temperatura; não prevejo melhora robusta de retorno."
    )
    predictions["part4_q"] = (
        "Ao usar apenas Q1 no alvo e na perda do ator, espero maior superestimação: "
        "Q - retorno soft tende a ficar mais positivo e Q1 pode crescer sem ganho "
        "proporcional no retorno. Espero maior instabilidade entre seeds e possível "
        "piora na avaliação, pois o ator pode explorar erros otimistas do crítico."
    )
    write(RESULTS / "predictions.json", predictions)
    print(f"Selected {final}; fixed alpha={alpha}. Predictions saved before Part 4.", flush=True)
    jobs = []
    for name in ["sac_hopper_fixed_alpha", "sac_hopper_single_q"]:
        jobs.append(subprocess.Popen(
            [sys.executable, "scripts/run_assignment.py", name, "--seeds", "1", "2"], cwd=ROOT))
    codes = [job.wait() for job in jobs]
    if any(codes):
        raise RuntimeError(f"Part 4 training failed: {codes}")
    run_script("export_assignment")
    run_script("analyze_assignment")
    print("All required experiments finished. Ready for interpretation and PDF.", flush=True)


if __name__ == "__main__":
    main()
