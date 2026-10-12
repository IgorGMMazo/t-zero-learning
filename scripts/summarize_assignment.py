"""Compact progress and reproducible statistics from the locally retained metrics."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def load_runs():
    runs = []
    for folder in sorted((ROOT / 'assignment_results').glob('*')):
        if not (folder / 'metadata.json').exists():
            continue
        meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        rows = []
        for line in (folder / 'history.jsonl').read_text(encoding='utf-8').splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        meta['rows'] = rows
        meta['completed'] = (folder / 'completed.json').exists()
        runs.append(meta)
    return runs


def summary(run):
    rows = run['rows']
    train = [r for r in rows if 'charts/SPS' in r]
    if not train:
        return {'id': run['id'], 'status': 'no training metrics'}
    last = train[-1]
    tail = train[-min(100, len(train)):]
    result = {'id': run['id'], 'config': run['config']['exp_name'],
              'seed': run['config']['seed'], 'completed': run['completed'],
              'step': last['global_step'],
              'return100': round(last.get('charts/episodic_return_mean_last100', 0), 2),
              'seconds': round(last['elapsed_seconds'], 1), 'SPS': last['charts/SPS']}
    for key in ['losses/approx_kl', 'losses/clipfrac', 'losses/entropy',
                'losses/explained_variance', 'losses/value_loss', 'charts/epochs_run']:
        values = [r[key] for r in tail if key in r]
        if values:
            result[key] = round(statistics.mean(values), 5)
    result['eval'] = next((r['eval/mean_return'] for r in reversed(rows) if 'eval/mean_return' in r), None)
    return result


if __name__ == '__main__':
    for run in load_runs():
        print(json.dumps(summary(run)))
