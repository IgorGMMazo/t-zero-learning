"""Render the four-page Portuguese report from measured statistics and figures."""
import argparse
import json
from pathlib import Path
import statistics
import yaml
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'reports'/'ppo'
OUT=ROOT.parent/'output'/'pdf'/'Relatorio_PPO_Igor_Christian.pdf'
W,H=595.28,841.89
M=38
CW=W-2*M
INK=HexColor('#172737')
BLUE=HexColor('#176ca4')
STYLE=ParagraphStyle('body',fontName='Helvetica',fontSize=9,leading=12,textColor=INK)


def number(x,d=1):
    return f'{x:,.{d}f}'.replace(',','X').replace('.',',').replace('X','.')


def flatten(d,p=''):
    out={}
    for k,v in d.items():
        key=f'{p}.{k}' if p else k
        if isinstance(v,dict): out.update(flatten(v,key))
        else: out[key]=v
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--final-config',default='ppo_lunarlander_final')
    args=parser.parse_args()
    data=json.loads((DATA/'statistics.json').read_text())
    runs=data['runs']
    initial=[r for r in runs if r['group']=='Inicial']
    ppo=[r for r in runs if r['group']=='PPO']
    a2c=[r for r in runs if r['group']=='A2C']
    OUT.parent.mkdir(parents=True,exist_ok=True)
    c=canvas.Canvas(str(OUT),pagesize=(W,H))
    c.setTitle('PPO e A2C no LunarLander-v3')
    c.setAuthor('Igor Garbin Manzan Mazo; Christian de Souza Ramos')
    y=0

    def p(text,size=9,gap=6):
        nonlocal y
        st=ParagraphStyle('p',parent=STYLE,fontSize=size,leading=size*1.3)
        para=Paragraph(text,st)
        _,height=para.wrap(CW,1000)
        para.drawOn(c,M,y-height)
        y-=height+gap
        assert y>38,('Page overflow',y,text[:80])

    def page(n,title):
        nonlocal y
        if n>1:c.showPage()
        c.setFillColor(BLUE);c.rect(M,H-37,27,3,fill=1,stroke=0)
        c.setFillColor(INK);c.setFont('Helvetica-Bold',17);c.drawString(M,H-65,title)
        c.setFont('Helvetica',8)
        c.drawString(M,23,'Igor Garbin Manzan Mazo | Christian de Souza Ramos')
        c.drawRightString(W-M,23,f'PPO / LunarLander-v3  -  {n}/4')
        y=H-82

    def fig(name,height,caption):
        nonlocal y
        c.drawImage(str(DATA/f'{name}.png'),M,y-height,width=CW,height=height,preserveAspectRatio=True,anchor='c')
        y-=height+4
        p(caption,7.5,9)

    def table(rows,widths=None):
        nonlocal y
        st=ParagraphStyle('cell',parent=STYLE,fontSize=7.7,leading=10)
        cells=[[Paragraph(str(x),st) for x in row] for row in rows]
        t=Table(cells,colWidths=widths or [CW/len(rows[0])]*len(rows[0]))
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),HexColor('#e5edf4')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),
            ('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),
            ('LINEBELOW',(0,0),(-1,0),.5,BLUE),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[white,HexColor('#f5f7f9')]),
        ]))
        _,hh=t.wrap(CW,1000);t.drawOn(c,M,y-hh);y-=hh+9
        assert y>38,('Table overflow',y)

    page(1,'PPO: resultado e protocolo experimental')
    solved=all(r['return100']>=200 and r['eval']>=200 for r in ppo)
    p('<b>Resultado:</b> '+('o PPO final atendeu aos dois critérios de retorno em ambas as seeds.' if solved else
      'o PPO testado ainda não atendeu integralmente ao critério em duas seeds.'))
    p('Implementamos GAE, a perda clipada com log-probabilidades antigas e vantagens constantes, e os diagnósticos sem gradiente. '
      'Os 34 testes de PPO/A2C passaram. A rede e as funções de A2C são as da atividade anterior; '
      'a busca alterou somente configurações de treino, sem modificar a dinâmica dos algoritmos.')
    p('Seeds 1 e 2; LunarLander-v3; MLPs separados de duas camadas de 64 unidades Tanh. '
      'PyTorch 2.9.1 CPU, Gymnasium 1.2.0 e Box2D 2.3.8, Windows. Avaliação final: dez episódios, '
      'política estocástica, conforme o executor do curso. A configuração inicial usa 1 milhão de passos; '
      'PPO final e A2C usam 2 milhões nominais ('+number(ppo[0]['step'],0)+' e '+number(a2c[0]['step'],0)+
      ' efetivos, respectivamente; diferença de arredondamento de '+number(abs(ppo[0]['step']-a2c[0]['step']),0)+' passos).')
    p('Tabela 1. Resultados ao final de cada treino.',8,4)
    rows=[['Configuração / seed','Run W&amp;B','Retorno últimos 100','Avaliação final']]
    for r in initial+ppo+a2c:
        rows.append([f"{r['group']} / {r['seed']}",r['id'],number(r['return100']),number(r['eval'])])
    table(rows,[113,95,150,161.28])
    fig('initial_final',210,'Figura 1. Retorno inicial, intermediário e final, duas seeds cada. Linhas tracejadas: seed 2; pontilhado horizontal: meta 200. Cada legenda identifica o run.')
    p('A configuração inicial não resolveu o ambiente: avaliações de '+
      ' e '.join(number(r['eval']) for r in initial)+'. Esse resultado e seus diagnósticos motivaram os ajustes da página seguinte.',8.5)

    page(2,'Ajustes orientados pelos diagnósticos')
    old=flatten(yaml.safe_load((ROOT/'configs'/'ppo_lunarlander.yml').read_text()))
    new=flatten(yaml.safe_load((ROOT/'configs'/f'{args.final_config}.yml').read_text()))
    rows=[['Chave','Inicial','Final']]
    for k,v in new.items():
        if old.get(k)!=v:rows.append([k,str(old.get(k)),str(v)])
    table(rows,[280,119,120.28])
    p('<b>Primeiro ajuste (Figura 2).</b> Nos runs iniciais 9lejwwgc e it98mxjn, a média dos últimos 100 registros '
      'de KL ficou próxima de 0,00001 e clipfrac foi zero; a taxa de aprendizado estava chegando a zero. '
      'A variância explicada ficou em 0,63 e 0,56. Testamos mais reuso (4 para 10 épocas), '
      'desativação do clipping da perda do crítico e maior orçamento, permitindo mais treino antes do fim do decaimento.',8)
    if data.get('trials'):
        p('<b>Segundo ajuste (Figuras 1 e 2).</b> Em hivjbi8g, o retorno caiu de cerca de 136 para 74; i0c05afs ficou perto de zero após 1 milhão. '
          'Entropias de 0,59/0,72 motivaram reduzir ent_coef. Rollouts de 128 para 1.024 e minibatches de 4 para 16 '
          '(nominalmente 128 para 256 amostras/minibatch) testam atualizações menos ruidosas. Gamma e lambda maiores ampliam o alcance '
          'dos retornos diante do progresso instável. Essas são hipóteses motivadas pelas curvas, não causas já demonstradas.',8)
    fig('diagnostics',300,'Figura 2. Inicial, intermediário e final (runs nas legendas); média móvel de 30 registros. Clipfrac mede razões fora do intervalo, não necessariamente gradientes efetivamente anulados.')
    p('<b>O que ocorreu depois.</b> No PPO final, a variância explicada média nos últimos 100 registros foi '+
      ' e '.join(number(r['losses/explained_variance'],2) for r in ppo)+
      '; entropias de '+ ' e '.join(number(r['losses/entropy'],2) for r in ppo)+
      '. As curvas estabilizaram acima de 200 (Figura 1); o KL volta a cair com o decaimento da taxa.',8)
    if data.get('trials'):
        p('Intermediário, treino/avaliação: '+ '; '.join(r['id']+': '+number(r['return100'])+'/'+number(r['eval']) for r in data['trials'])+
          '. As alterações foram conjuntas: a comparação não isola a contribuição causal de cada uma.',8)

    page(3,'PPO versus A2C: amostras e tempo')
    p('A2C partiu de a2c_cartpole, com LunarLander-v3, o mesmo orçamento de 2 milhões e num_steps=32 em vez de 5. '
      'O retorno mais longo busca representar melhor consequências do pouso; os demais hiperparâmetros foram preservados. '
      'É uma tentativa de comparação, sem exigência de resolver o ambiente com A2C.',8.5)
    fig('comparison',307,'Figura 3. Mesmas seeds e orçamento: retorno × passos, retorno × tempo, SPS e número acumulado de passos de gradiente. Tempo contado após a inicialização do W&B; avaliação e upload ficam fora das curvas.')
    rows=[['Run','Passos até 0','Até 100','Até 200','Tempo / SPS']]
    for r in ppo+a2c:
        vals=[]
        for threshold in ['0','100','200']:
            cross=r['crossings'][threshold]
            vals.append('não atingiu' if cross is None else number(cross['step'],0))
        rows.append([f"{r['group']} s{r['seed']}",*vals,f"{number(r['seconds']/60)} min / {r['SPS']}"])
    table(rows,[76,105,105,105,128.28])
    p('Os limiares usam a primeira média dos últimos 100 episódios registrada acima do valor; não significam '
      'permanência. No orçamento final, PPO executou '+number(ppo[0]['num_updates'],0)+
      ' atualizações por seed, contra '+number(a2c[0]['num_updates'],0)+' no A2C. '
      'O reuso de minibatches aumenta o custo de otimização por passo de ambiente e ajuda a explicar o menor SPS do PPO.',8.5)
    bygroup={g:statistics.mean(r['return'] for r in data['same_time'] if r['group']==g) for g in ['PPO','A2C']}
    p('Com o mesmo tempo observado de '+number(data['common_time_seconds']/60)+' minutos, a média entre seeds foi '+
      number(bygroup['PPO'])+' para PPO e '+number(bygroup['A2C'])+' para A2C. '
      'Essa leitura usa o último registro de cada run disponível até esse instante. Execução concorrente e comprimentos de episódio '
      'diferentes limitam a generalização dos tempos.',8.5)

    page(4,'Estabilidade, mecanismos e rastreabilidade')
    fig('stability',175,'Figura 4. Entropia e variância explicada de PPO e A2C, com os runs identificados; média móvel de 30 registros. Interpretar junto do retorno na Figura 3.')
    gap_ppo=abs(ppo[0]['return100']-ppo[1]['return100'])
    gap_a2c=abs(a2c[0]['return100']-a2c[1]['return100'])
    p('<b>Estabilidade.</b> A diferença final entre seeds foi '+number(gap_ppo)+' pontos no PPO e '+number(gap_a2c)+
      ' no A2C. A maior queda entre um pico anterior e um registro posterior foi '+
      ' / '.join(number(r['max_drawdown']) for r in ppo)+' no PPO e '+
      ' / '.join(number(r['max_drawdown']) for r in a2c)+' no A2C (seeds 1 / 2). '
      'São estatísticas descritivas das curvas de treino, não intervalos de confiança; duas seeds não bastam para afirmar superioridade universal.',8.5)
    p('<b>GAE.</b> Combina resíduos temporais com fator gamma * lambda, enquanto o A2C usa um único retorno de n passos. '
      'As curvas do crítico (Figura 4) são compatíveis com diferenças na qualidade dos alvos e vantagens, mas também dependem '
      'do ajuste de valor e do número de atualizações. Não há uma ablação que isole GAE.',8.5)
    p('<b>Reuso de amostras.</b> A Figura 3 mostra diretamente o maior número de atualizações por amostra no PPO. '
      'Isso permite extrair mais aprendizado de cada rollout, com custo de tempo adicional. A curva de retorno por passos '
      'é a medida de eficiência amostral; SPS sozinho não mede qualidade.',8.5)
    p('<b>Razão de probabilidades.</b> Corrige o objetivo para a política que coletou o rollout, necessária quando os mesmos '
      'dados são reutilizados após atualizações. O KL da Figura 2 mede esse afastamento; não há uma razão equivalente no '
      'único update on-policy de A2C.',8.5)
    p('<b>Clipping.</b> Limita o incentivo do objetivo a mudanças excessivas na direção favorável à vantagem. '
      'A clipfrac da Figura 2 quantifica quantas razões saem do intervalo; não garante um limite rígido de KL nem ausência de '
      'quedas de retorno. Seu papel deve ser interpretado junto do reuso e da correção por razões.',8.5)
    p('<b>Código e evidências.</b> <link href="https://github.com/IgorGMMazo/t-zero-learning/tree/codex/ppo-assignment" color="#176ca4">'
      'Fork: IgorGMMazo/t-zero-learning, branch codex/ppo-assignment</link>. '
      'Implementação, configs, scripts de análise e dados em reports/ppo. '
      '<link href="https://wandb.ai/igor_mazo-federal-university-of-goi-s/ppo-assignment" color="#176ca4">Projeto W&amp;B: ppo-assignment</link>.',8)
    links=['<link href="'+r['url']+'" color="#176ca4">'+r['group']+' s'+str(r['seed'])+': '+r['id']+'</link>' for r in initial+ppo+a2c]
    p('Runs: '+'; '.join(links)+'.',7.5)
    c.save()
    assert len(PdfReader(OUT).pages)==4
    print(OUT)


if __name__=='__main__':
    main()
