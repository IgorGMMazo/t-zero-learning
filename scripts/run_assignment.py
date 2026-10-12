"""Run the unchanged course trainer, also retaining its metrics as local JSONL.

Usage: python scripts/run_assignment.py --config ppo_lunarlander --override seed=1
"""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    os.chdir(ROOT)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ['WANDB_PROJECT'] = 'ppo-assignment'
    import torch
    import wandb
    import train

    torch.set_num_threads(1)
    original_init = wandb.init
    state = {}

    def init(*args, **kwargs):
        run = original_init(*args, **kwargs)
        folder = ROOT / 'assignment_results' / run.id
        folder.mkdir(parents=True, exist_ok=True)
        state.update(folder=folder, start=time.perf_counter(), config=kwargs.get('config', {}))
        (folder / 'metadata.json').write_text(json.dumps({
            'id': run.id, 'name': run.name, 'url': run.url,
            'config': state['config'], 'command': sys.argv,
            'torch_threads': torch.get_num_threads(),
        }, indent=2), encoding='utf-8')
        state['stream'] = (folder / 'history.jsonl').open('a', encoding='utf-8', buffering=1)
        # The SDK replaces module-level log on init; install the tee afterwards.
        state['sdk_log'] = wandb.log
        wandb.log = log
        return run

    def log(data, *args, **kwargs):
        row = dict(data)
        row['elapsed_seconds'] = time.perf_counter() - state['start']
        if 'step' in kwargs:
            row.setdefault('global_step', kwargs['step'])
        cfg = state['config']
        if cfg.get('algorithm') == 'a2c' and 'global_step' in row:
            row['charts/num_updates'] = row['global_step'] // cfg['batch_size']
        state['stream'].write(json.dumps(row, default=lambda x: x.item()) + '\n')
        return state['sdk_log'](row, *args, **kwargs)

    wandb.init = init
    try:
        train.main()
        if state:
            (state['folder'] / 'completed.json').write_text(json.dumps({
                'elapsed_seconds': time.perf_counter() - state['start']
            }), encoding='utf-8')
    finally:
        if 'stream' in state:
            state['stream'].close()
        wandb.finish()


if __name__ == '__main__':
    main()
