"""Build report figures and statistics from actual completed runs, never mock data."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from summarize_assignment import load_runs, summary

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'ppo'
RETURN = 'charts/episodic_return_mean_last100'
COLORS = {'Inicial': '#929eac', 'PPO': '#176ca4', 'A2C': '#ce6b28', 'Interm.': '#9462a6'}


def prepare(final_config='ppo_lunarlander_final'):
    runs = [r for r in load_runs() if r['completed'] and r['config']['exp_name'] in
            ('ppo_lunarlander', final_config, 'a2c_lunarlander')]
    for r in runs:
        name = r['config']['exp_name']
        r['group'] = 'Inicial' if name == 'ppo_lunarlander' else ('A2C' if name.startswith('a2c') else 'PPO')
        r['label'] = f"{r['group']} s{r['config']['seed']} ({r['id']})"
        r['train'] = [x for x in r['rows'] if RETURN in x]
    return sorted(runs, key=lambda r: (r['group'], r['config']['seed']))


def plot(ax, runs, key, xkey='global_step', smooth=1):
    for r in runs:
        rows = [x for x in r['train'] if key in x]
        if not rows:
            continue
        x = np.array([x[xkey] for x in rows]) / (1e6 if xkey == 'global_step' else 60)
        y = np.array([x[key] for x in rows], dtype=float)
        if smooth > 1:
            y = np.array([np.nanmean(y[max(0, i-smooth+1):i+1]) for i in range(len(y))])
        ax.plot(x, y, color=COLORS[r['group']], ls='-' if r['config']['seed']==1 else '--',
                lw=1.2, alpha=.92, label=r['label'])
    ax.grid(alpha=.18)
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlabel('Passos de ambiente (milhões)' if xkey=='global_step' else 'Tempo decorrido (min)')
    if key==RETURN:
        ax.axhline(200, color='#333333', ls=':', lw=.8)
        ax.set_ylabel('Retorno médio / últimos 100 episódios')


def save(fig,name):
    fig.savefig(OUT / f'{name}.png', dpi=190, bbox_inches='tight')
    plt.close(fig)


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--final-config',default='ppo_lunarlander_final')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':8,'axes.titlesize':9,'legend.fontsize':6.5,'font.family':'DejaVu Sans'})
    runs=prepare(args.final_config)
    if len(runs)!=6:
        raise RuntimeError(f'Expected 6 completed comparison runs; found {len(runs)}')
    ppo=[r for r in runs if r['group']!='A2C']
    comp=[r for r in runs if r['group']!='Inicial']
    trials=[]
    for r in load_runs():
        if r['completed'] and r['config']['exp_name']=='ppo_lunarlander_tuned' and args.final_config!='ppo_lunarlander_tuned':
            r.update(group='Interm.',label=f"Interm. s{r['config']['seed']} ({r['id']})",
                     train=[x for x in r['rows'] if RETURN in x])
            trials.append(r)
    fig,ax=plt.subplots(figsize=(7.4,2.8),layout='constrained')
    plot(ax,ppo+trials,RETURN);ax.legend(loc='lower right',ncol=2);save(fig,'initial_final')
    fig,axs=plt.subplots(3,2,figsize=(7.4,5.6),layout='constrained')
    for ax,key,title in zip(axs.flat,['losses/approx_kl','losses/clipfrac','losses/explained_variance','losses/value_loss',
                                    'losses/entropy','charts/learning_rate'],
                            ['KL aproximado','Fração fora do intervalo de clipping','Variância explicada do crítico','Perda do crítico',
                             'Entropia da política','Taxa de aprendizado']):
        plot(ax,ppo+trials,key,smooth=30);ax.set_title(title)
        if key=='losses/value_loss':
            ax.set_yscale('log')
            ax.set_title(title+' (escala log)')
    axs[0,0].legend();save(fig,'diagnostics')
    fig,axs=plt.subplots(2,2,figsize=(7.4,4.3),layout='constrained')
    plot(axs[0,0],comp,RETURN);plot(axs[0,1],comp,RETURN,xkey='elapsed_seconds')
    plot(axs[1,0],comp,'charts/SPS');axs[1,0].set_ylabel('Passos por segundo (acumulado)')
    plot(axs[1,1],comp,'charts/num_updates');axs[1,1].set_ylabel('Passos de gradiente acumulados')
    axs[0,0].legend();save(fig,'comparison')
    fig,axs=plt.subplots(1,2,figsize=(7.4,2.4),layout='constrained')
    plot(axs[0],comp,'losses/entropy',smooth=30);axs[0].set_ylabel('Entropia da política')
    plot(axs[1],comp,'losses/explained_variance',smooth=30);axs[1].set_ylabel('Variância explicada')
    axs[0].legend();save(fig,'stability')
    stats=[]
    for r in runs:
        s=summary(r);s.update(group=r['group'],url=r['url'],label=r['label'],
                             return100=r['train'][-1][RETURN],seconds=r['train'][-1]['elapsed_seconds'])
        s['num_updates']=r['train'][-1]['charts/num_updates']
        s['crossings']={}
        for threshold in [0,100,200]:
            crossing=next((x for x in r['train'] if x[RETURN]>=threshold),None)
            s['crossings'][str(threshold)]=None if crossing is None else {
                'step':crossing['global_step'],'seconds':crossing['elapsed_seconds']}
        returns=np.array([x[RETURN] for x in r['train']])
        s['max_drawdown']=float(np.max(np.maximum.accumulate(returns)-returns))
        s['peak_return']=float(np.max(returns))
        lastquarter=[x[RETURN] for x in r['train'] if x['global_step']>=.75*r['train'][-1]['global_step']]
        s['lastquarter_min']=float(np.min(lastquarter))
        s['lastquarter_max']=float(np.max(lastquarter))
        stats.append(s)
    horizon=min(r['train'][-1]['elapsed_seconds'] for r in comp)
    same_time=[]
    for r in comp:
        row=max((x for x in r['train'] if x['elapsed_seconds']<=horizon),key=lambda x:x['elapsed_seconds'])
        same_time.append({'id':r['id'],'group':r['group'],'step':row['global_step'],'return':row[RETURN]})
    (OUT/'statistics.json').write_text(json.dumps({'runs':stats,'trials':[summary(r) for r in trials],
        'common_time_seconds':horizon,'same_time':same_time},indent=2),encoding='utf-8')
    print(json.dumps(stats,indent=2))


if __name__=='__main__':
    main()
