# t-zero

## SAC assignment

The SAC solution is on branch `codex/sac-assignment`. The original course
implementation is preserved except for the two requested SAC functions and
a `train.py` fix that prints SAC's separate actor and critic learning rates.

```powershell
python -m pytest tests/test_sac.py
python scripts/run_assignment.py sac_pendulum --seeds 1
python scripts/run_assignment.py sac_hopper
python scripts/run_assignment.py sac_hopper_early
python scripts/run_assignment.py sac_hopper_slow_q
python scripts/run_assignment.py sac_hopper_utd2
python scripts/run_assignment.py ppo_pendulum --seeds 1 2 --override env_id=Hopper-v5 total_timesteps=100000 ppo_continuous_action.gamma=0.99 track=true
```

Set your own W&B credentials in `.env` and `WANDB_PROJECT=sac-assignment`.
The runner sets one CPU thread per process and uses UTF-8 logs on Windows.
For this local workspace it can also read credentials from the sibling
`t-zero-learning/.env`; credentials are never included in the report or Git.
The original `python train.py --config ...` entry point remains usable.
For Linux headless MuJoCo video, set `MUJOCO_GL=egl`.

Experiment evidence is in `results/`: candidate rationales, predictions
written before ablations/comparison, environment versions, seed summaries,
and exported W&B histories. `scripts/export_assignment.py` reads complete
local W&B event files without sampling; `--remote` downloads through the
W&B API instead. Only finished runs with final evaluations are exported
by the local reader. The final report is generated with
`python scripts/build_report.py` once all required completed runs and
`results/conclusions.json` are present. It refuses incomplete experiment
groups and PDFs exceeding four pages. Install the report dependencies with
`python -m pip install -r requirements-sac-report.txt`.

The supplied PPO command configures 100,000 steps but the unchanged PPO
loop runs 24 full rollouts of 4,096 steps (98,304 actual steps). The report
must disclose this difference. SAC runs exactly 100,000 loop steps.

`python scripts/complete_experiments.py` waits for the twelve Part 3 runs
started by the runner, applies the recorded conservative selection rule,
writes the Part 4 predictions and configs, runs both ablations with two
seeds, and exports the completed evidence. It stops on a failed run.
It does not synthesize conclusions or invent missing results.

To rebuild the submitted report from the public experiment project:

```powershell
python scripts/export_assignment.py --remote --finished-only
python scripts/analyze_assignment.py
python scripts/diagnostics.py
python scripts/validate_assignment.py
python scripts/build_report.py
```

The final PDF and figures are in `output/pdf/`. Full histories and training
logs stay local; compact summaries, predictions, and configuration evidence
are versioned in `results/`.

A readable starting point for reinforcement-learning research, built on
[CleanRL](https://github.com/vwxyzjn/cleanrl)'s single-file implementations.

**t-zero is not a library.** It is a codebase you clone, read, and modify. It is
meant as the *time step zero* of your RL project: a working, tested foundation
where developers, researchers, and students apply RL to **new environments,
experiments, and algorithms**. Our goal is to give full control over, and clear visibility into everything that happens. Nothing is hidden behind a convenient function call.

## Built on CleanRL

The algorithms are taken directly from **CleanRL**
([Huang et al., 2022](https://www.jmlr.org/papers/v23/21-1342.html)), for
example, the training loop, network initialization, and all code-level
optimizations in
[algorithms/ppo_continuous_action.py](algorithms/ppo_continuous_action.py)
follow CleanRL's `ppo_continuous_action.py` (see
[The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)).
What t-zero adds is the scaffolding around the algorithm: YAML configs with
strict validation, an environment factory with adapters and wrapper stacks,
checkpoint/resume semantics, evaluation and video tooling, cluster scripts,
and a test suite that picks up your extensions automatically.

## Quickstart

```bash
pip install -r requirements.txt
python train.py --config ppo_pendulum      # ~3 minutes on CPU
```

Any config field can be overridden from the CLI, and typos abort at startup
instead of silently training with defaults:

```bash
python train.py --config ppo_halfcheetah --override seed=7 algo.gamma=0.95
```

Interrupt and resume any run:

```bash
python train.py --resume runs/ppo_pendulum/<run_dir>
```

Evaluate and record videos:

```bash
MUJOCO_GL=egl python evaluate.py runs/ppo_pendulum/<run_dir> --deterministic
```

The full walkthrough: install, what lands on disk, overrides, resume,
evaluation are in [docs/quickstart-training.md](docs/quickstart-training.md).

## What's inside

| | |
| --- | --- |
| [algorithms/](algorithms/) | Algorithm implementations, new algorithms register here |
| [configs/](configs/) | Benchmark and example configs (see below) |
| [envs/](envs/) | Env factory, wrapper stacks, adapters (Meta-World), custom envs (AntDir, AntDirGoal, HalfCheetahVel, HalfCheetahVelGoal) |
| [networks/](networks/) | Agent network architectures |
| [core/](core/) | Config dataclasses, YAML loader/overrides, checkpointing, run naming |
| [train.py](train.py) / [evaluate.py](evaluate.py) | Entry points |
| [tests/](tests/) | Contract + smoke tests (`pytest`, ~4s fast tier) |
| [docker/](docker/) / [cluster/](cluster/) | Docker Compose setup and SLURM/Apptainer scripts |



## Configs: benchmarks and examples

Configs are complete, runnable experiments; the name of the config becomes the
experiment name.

**CleanRL benchmarks** — hyperparameters match CleanRL's
`ppo_continuous_action.py` benchmark suite (MuJoCo v4, 1M steps), so results
can be checked against the
[published reference returns](https://docs.cleanrl.dev/rl-algorithms/ppo/#experiment-results_2)
(noted in each file's header):

- [configs/ppo_halfcheetah.yml](configs/ppo_halfcheetah.yml)
- [configs/ppo_hopper.yml](configs/ppo_hopper.yml)
- [configs/ppo_walker2d.yml](configs/ppo_walker2d.yml)
- [configs/ppo_humanoid.yml](configs/ppo_humanoid.yml)

**Examples** — each demonstrates one capability of the scaffolding:

- [configs/ppo_pendulum.yml](configs/ppo_pendulum.yml) — commented quickstart, minutes on CPU
- [configs/ppo_halfcheetahvel.yml](configs/ppo_halfcheetahvel.yml) — custom env with `env_kwargs`
- [configs/ppo_antdir.yml](configs/ppo_antdir.yml) — custom env, vectorized GPU training
- [configs/ppo_antdir_goal.yml](configs/ppo_antdir_goal.yml) — non-stationary goals (mid-episode switches)
- [configs/ppo_metaworld_mt10.yml](configs/ppo_metaworld_mt10.yml) — multi-task benchmark via an env adapter + split-optimizer PPO

## Documentation

Start with the quickstart, then read the guide that matches what you want to
change:

- [docs/quickstart-training.md](docs/quickstart-training.md) — fresh clone to trained, evaluated policy
- [docs/config-reference.md](docs/config-reference.md) — every config key (generated from the dataclasses)
- [docs/adding-a-new-environment.md](docs/adding-a-new-environment.md) — from stock Gymnasium ids to custom envs with their own vectoriser
- [docs/adding-a-new-algorithm.md](docs/adding-a-new-algorithm.md) — the `Algorithm` lifecycle and registration
- [docs/adding-metrics.md](docs/adding-metrics.md) — logging new quantities
- [docs/checkpoint-semantics.md](docs/checkpoint-semantics.md) — exactly what resume does and doesn't restore
- [docs/arena.md](docs/arena.md) — GFootball arena: match setup, anchors, check / match / tournament options
- [docs/testing.md](docs/testing.md) — what the suite guarantees and what you owe it when extending
- [docs/t-zero-design-principles.md](docs/t-zero-design-principles.md) — design principles

## Running with Docker

```bash
docker-compose run --rm t-zero python train.py --config ppo_halfcheetah
```

Run from the repo root (not from inside the devcontainer) so the volume
mapping keeps results on the host.

## Running on a SLURM cluster

[cluster/](cluster/) contains an Apptainer/Singularity definition and sbatch
scripts:

```bash
apptainer build --fakeroot t-zero.sif cluster/t-zero.def
sbatch --export=ALL,WORKDIR=/path/to/t-zero,CONFIG=ppo_antdir_goal cluster/run_seeds_parallel.sh
```

`scripts/run_seeds.sh` runs one config for multiple seeds (sequentially or in
parallel) on any machine; the cluster scripts wrap it for SLURM. Adjust the
`#SBATCH --partition=...` lines to your cluster.

## Testing

```bash
pytest                  # full suite (~4s fast tier + tiny CPU training runs)
pytest -m "not slow"    # fast tier only
```

Extensions are tested automatically: registering a config, algorithm, wrapper
stack, or custom env opts it into the corresponding contract tests. See
[docs/testing.md](docs/testing.md).
