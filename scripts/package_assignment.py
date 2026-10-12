"""Archive only completed experiment evidence, plus reproducibility metadata."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import zipfile
from summarize_assignment import load_runs

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'/'ppo'


def main():
    runs=[r for r in load_runs() if r['completed']]
    OUT.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(OUT/'experiment_data.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for r in runs:
            folder=ROOT/'assignment_results'/r['id']
            for name in ('metadata.json','history.jsonl','completed.json'):
                z.write(folder/name,arcname=f"assignment_results/{r['id']}/{name}")
        decisions=ROOT/'assignment_results'/'decisions.json'
        z.write(decisions,arcname='assignment_results/decisions.json')
    import gymnasium,torch,Box2D,wandb
    files=['algorithms/ppo.py','algorithms/a2c.py','networks/discrete_actor_critic.py']
    manifest={'python':platform.python_version(),'os':platform.platform(),'processor':platform.processor(),
        'versions':{'torch':torch.__version__,'gymnasium':gymnasium.__version__,
                    'Box2D':Box2D.__version__,'wandb':wandb.__version__},
        'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
        'upstream_commit':subprocess.check_output(['git','rev-parse','upstream/main'],cwd=ROOT,text=True).strip(),
        'runs':[{'id':r['id'],'url':r['url'],'config':r['config']['exp_name'],'seed':r['config']['seed']} for r in runs],
        'excluded_infrastructure_failure':'09uq3wmc (local JSON serialization error before any metric; rerun from scratch)',
        'timing_note':'Runs overlapped; one PyTorch/OpenMP/MKL thread per trainer, AsyncVectorEnv workers. Times are not isolated CPU benchmarks.'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Archived {len(runs)} completed runs')


if __name__=='__main__':
    main()
