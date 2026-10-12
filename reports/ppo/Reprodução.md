# Experimentos de PPO e A2C no LunarLander

Igor Garbin Manzan Mazo e Christian de Souza Ramos.

## Ambiente

Windows, Python 3.11, PyTorch 2.9.1 CPU, Gymnasium 1.2.0, Box2D-py 2.3.8.
Instale as dependências do projeto e as de `requirements-ppo-extra.txt`.
No Windows, instale `swig` e `wheel` antes de `box2d-py`, coloque a pasta
`Scripts` do ambiente virtual no PATH e use `--no-build-isolation` para Box2D.

O arquivo `.env` mantém as credenciais locais. O executor usa explicitamente
`WANDB_PROJECT=ppo-assignment`, sem imprimir credenciais nem alterar outros
projetos do W&B. Uma thread PyTorch/OpenMP/MKL por processo de treino.

## Testes

```powershell
python -m pytest tests/test_ppo.py tests/test_a2c.py
```

As 34 verificações das duas atividades passaram. A suíte completa teve 150
testes aprovados, três pulados e três falhas relacionadas à ausência de DQN em
`tests/helpers.py::SMOKE_SETTINGS`, não ao PPO/A2C. Os testes de treino e retomada
de PPO/A2C passaram. O gerador da referência de configuração foi corrigido para
usar UTF-8 e caminhos com `/` também no Windows.

## Experimentos

Execute cada configuração para `seed=1` e `seed=2`, a partir da raiz:

```powershell
python scripts/run_assignment.py --config ppo_lunarlander --override seed=1
python scripts/run_assignment.py --config ppo_lunarlander_tuned --override seed=1
python scripts/run_assignment.py --config ppo_lunarlander_final --override seed=1
python scripts/run_assignment.py --config a2c_lunarlander --override seed=1
```

O executor chama o `train.py` do curso sem alterar as atualizações dos algoritmos.
Ele copia as métricas reais para `assignment_results/<run-id>/history.jsonl` e
salva configuração, URL, comando e estado de conclusão. Para A2C, registra também
`charts/num_updates = global_step / batch_size`, pois há exatamente um passo de
gradiente por rollout. O contador do PPO já é fornecido pelo algoritmo.

A configuração inicial usa 1.000.000 de passos nominais (999.936 executados).
PPO intermediário e A2C usam 2.000.000 (1.999.872 executados em ambos).
PPO final também usa 2.000.000 nominais, mas executa 1.998.848, pois o rollout
é maior. A diferença de 1.024 passos (0,0512% do orçamento) é declarada no relatório;
decorre do arredondamento para rollouts completos feito pelo código oficial.

A2C preserva as implementações da atividade anterior. Parte de `a2c_cartpole`,
muda o ambiente, o orçamento e `a2c.num_steps` de 5 para 32: alvo de retorno mais
longo para o pouso, mantendo learning rate, entropia, rede e demais parâmetros.
Não se exige que A2C resolva o ambiente.

## Interpretação e limites

- A avaliação final segue o curso: dez episódios com ações amostradas da política.
- Retorno de treino é a média móvel dos últimos 100 episódios, sem suavização extra.
- Diagnósticos nas figuras usam média móvel de 30 registros, declarada nas legendas.
- Cruzamentos de 0, 100 e 200 são a primeira observação registrada acima do limiar;
  não implicam que o resultado permaneceu acima dele.
- Tempos são medidas desta máquina sob execução concorrente. Incluem pequena
  inicialização após `wandb.init`; a tabela principal exclui a avaliação e o upload.
  Não são benchmarks isolados de CPU nem estimativas universais.
- As alterações do PPO foram testadas em conjunto: os gráficos sustentam hipóteses,
  mas não isolam causalmente cada mecanismo. Duas seeds não são uma estimativa
  precisa da distribuição de desempenho.

## Análise

```powershell
python scripts/summarize_assignment.py
python scripts/plot_assignment.py
python scripts/build_assignment_report.py
```

As figuras e `statistics.json` são derivados dos logs, sem resultados simulados.
Os identificadores e links dos runs ficam no relatório e nos arquivos de evidência.

O arquivo `experiment_data.zip` contém os oito treinos concluídos e o registro das
decisões experimentais. Para refazer as figuras em um clone novo, extraia-o na raiz:

```powershell
python -m zipfile -e reports/ppo/experiment_data.zip .
python scripts/plot_assignment.py
python scripts/build_assignment_report.py
```

O PDF gerado vai para `../output/pdf/Relatorio_PPO_Igor_Christian.pdf`. A cópia
entregue também está nesta pasta. A validação final das funções, configurações
e referência de configuração aprovou 45 testes selecionados. Os pesos treinados
e vídeos permanecem nos diretórios locais `runs/`; o arquivo de evidências guarda
as métricas completas, configurações e links do W&B, sem credenciais.
