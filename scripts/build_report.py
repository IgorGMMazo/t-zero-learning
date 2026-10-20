"""Build the four-page report only after completed histories and conclusions exist."""
import json
from pathlib import Path
import sys
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf"
PROJECT = "https://wandb.ai/igor_mazo-federal-university-of-goi-s/sac-assignment"
COLORS = ["#2563eb", "#ea580c"]


def main():
    conclusions = json.loads((ROOT / "results" / "conclusions.json").read_text(encoding="utf-8"))
    if any("PENDENTE" in str(value) for value in conclusions.values()):
        raise ValueError("Finish interpreting all experiments before generating the report.")
    aggregates = json.loads((ROOT / "results" / "aggregates.json").read_text(encoding="utf-8"))
    rationale = json.loads((ROOT / "results" / "candidate_rationale.json").read_text(encoding="utf-8"))
    runs = {}
    for path in (ROOT / "results" / "histories").glob("*.json"):
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["state"] == "finished" and "eval/mean_return" in run["summary"]:
            runs.setdefault(run["config"]["exp_name"], []).append(run)
    for group in runs.values():
        group.sort(key=lambda r: r["config"]["seed"])
    final = conclusions["final_config"]
    competitor = conclusions["comparison_config"]
    required = {"sac_hopper": 3, **{r["config"]: 3 for r in rationale},
                "sac_hopper_fixed_alpha": 2, "sac_hopper_single_q": 2, "ppo_pendulum": 2}
    for name, count in required.items():
        if len(runs.get(name, [])) != count:
            raise ValueError(f"Expected {count} finished seeds for {name}")
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 160})

    def figure(filename, configs, keys, labels, time_axis=False, nseeds=None):
        if time_axis and len(keys) == 1:
            keys = keys * 2
            labels = labels * 2
        fig, axes = plt.subplots(1, len(keys), figsize=(7.1, 2.25), squeeze=False)
        for panel, (ax, key, label) in enumerate(zip(axes[0], keys, labels)):
            for j, config in enumerate(configs):
                group = runs[config][:nseeds] if nseeds else runs[config]
                for i, run in enumerate(group):
                    pairs = [(r.get("_runtime") if time_axis else r.get("global_step"), r[key])
                             for r in run["history"] if key in r]
                    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
                    if not pairs:
                        raise ValueError(f"No history: {run['id']} {key}")
                    x, y = map(np.asarray, zip(*pairs))
                    if time_axis:
                        y = y.astype(float)
                        y[1:][np.diff(x) > 60] = np.nan
                    # Unsmoothed histories preserve the seed-level variability.
                    ax.plot(x / (60 if time_axis else 1000), y, color=COLORS[j],
                            linewidth=.85, alpha=.78, linestyle=["-", "--", ":"][i],
                            label=f"{conclusions['labels'][config]} s{run['config']['seed']}")
            if key == "losses/entropy":
                ax.axhline(-3, color="#555555", linestyle=":", linewidth=1, label="Alvo = -3")
                ax.text(.98, -3, "alvo -3", transform=ax.get_yaxis_transform(),
                        ha="right", va="bottom", fontsize=6)
            if key == "charts/q_minus_return":
                ax.axhline(0, color="#777777", linewidth=.6)
            ax.set(xlabel="Tempo (min)" if time_axis else "Passos (mil)", ylabel=label)
            if time_axis and panel == 1:
                common_end = min(
                    max(r["_runtime"] for r in run["history"] if key in r) / 60
                    for run in runs[configs[1]][:nseeds]
                )
                ax.set_xlim(0, common_end)
                common_values = [
                    r[key] for config in configs for run in runs[config][:nseeds]
                    for r in run["history"]
                    if key in r and r.get("_runtime", float("inf")) <= common_end * 60
                ]
                ax.set_ylim(min(0, min(common_values)), max(common_values) * 1.1)
                ax.set_title("Janela comum: até o fim do PPO", fontsize=8)
            elif time_axis:
                ax.set_title("Tempo total registrado", fontsize=8)
            ax.grid(alpha=.15)
        axes[0][0].legend(fontsize=6, loc="best")
        fig.tight_layout(pad=.8)
        path = OUT / filename
        fig.savefig(path, dpi=190)
        plt.close(fig)
        return path

    p3 = figure("eficiencia.png", ["sac_hopper", competitor],
                ["charts/episodic_return_mean_last100"], ["Retorno (últimos 100)"])
    p4a = figure("temperatura.png", [final, "sac_hopper_fixed_alpha"],
                ["losses/alpha", "losses/entropy", "charts/episodic_return_mean_last100"],
                ["Temperatura", "Entropia estimada", "Retorno (últimos 100)"], nseeds=2)
    p4b = figure("criticos.png", [final, "sac_hopper_single_q"],
                ["charts/q_minus_return", "losses/qf1_values", "charts/episodic_return_mean_last100"],
                ["Q - retorno soft", "Q1 médio", "Retorno (últimos 100)"], nseeds=2)
    p5a = figure("sac_ppo_passos.png", [final, "ppo_pendulum"],
                ["charts/episodic_return_mean_last100"], ["Retorno (últimos 100)"], nseeds=2)
    p5b = figure("sac_ppo_tempo.png", [final, "ppo_pendulum"],
                ["charts/episodic_return_mean_last100"], ["Retorno (últimos 100)"],
                time_axis=True, nseeds=2)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodySmall", fontName="Helvetica", fontSize=9,
                              leading=12, spaceAfter=7, textColor=colors.HexColor("#243247")))
    styles.add(ParagraphStyle(name="CaptionSmall", fontSize=7, leading=9,
                              spaceAfter=8, textColor=colors.HexColor("#526174")))
    styles.add(ParagraphStyle(name="CellSmall", fontSize=7, leading=9))
    styles["Title"].fontSize = 20
    styles["Title"].leading = 23
    styles["Title"].textColor = colors.HexColor("#17365d")
    story = []

    def p(text, style="BodySmall"):
        story.append(Paragraph(text, styles[style]))

    def pic(path):
        story.append(Image(str(path), width=503, height=159))

    def ids(config, count=None):
        group = runs[config][:count] if count else runs[config]
        return ", ".join(f'<a href="{r["url"]}" color="#2563eb">{r["id"]}</a> (s{r["config"]["seed"]})' for r in group)

    def stats(config, key, count=None):
        if count:
            values = [r["summary"][key] for r in runs[config][:count]]
            return f"{statistics.mean(values):.1f} ({min(values):.1f} a {max(values):.1f})"
        s = aggregates[config][key]
        return f'{s["mean"]:.1f} ({s["min"]:.1f} a {s["max"]:.1f})'

    p("Soft Actor-Critic no Hopper-v5", "Title")
    p(conclusions["author"] + " | Atividade SAC | 20/10/2026")
    p(f'<a href="{conclusions["fork_url"]}" color="#2563eb">Código e configurações no fork</a> | '
      f'<a href="{PROJECT}" color="#2563eb">Projeto W&amp;B e runs</a>', "CaptionSmall")
    p("<b>Validação.</b> Reparametrização com rsample(), correção do jacobiano do tanh e alvo soft implementados. "
      "Os 16 testes SAC passaram. Pendulum: retorno determinístico -132,9 em 10 episódios (limiar: -200). "
      "Apenas os dois blocos do SAC foram alterados; train.py recebeu uma correção na impressão das taxas.")
    p("<b>Parte 3 - Eficiência amostral.</b> Cada configuração SAC usou exatamente 100.000 passos e seeds 1, 2 e 3. "
      "Valores: média (mínimo a máximo). Os intervalos são amplitudes entre seeds, não intervalos de confiança.")
    rows = [["Mudança / motivo observado", "Avaliação", "Últimos 100", "Veredito"]]
    rows.append(["Inicial (CleanRL)", stats("sac_hopper", METRIC_EVAL),
                 stats("sac_hopper", METRIC_TRAIN), "Referência"])
    for item in rationale:
        name = item["config"]
        rows.append([f'{item["key"]}: {item["old"]} → {item["new"]}<br/>{item["reason"]}',
                     stats(name, METRIC_EVAL), stats(name, METRIC_TRAIN),
                     conclusions["verdicts"][name]])
    table = Table([[Paragraph(str(c).replace("→", "-&gt;"), styles["CellSmall"]) for c in row] for row in rows],
                  colWidths=[239, 91, 91, 82])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5edf7")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), .3, colors.HexColor("#cdd7e4")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([table, Spacer(1, 8)])
    p(conclusions["selection"])
    pic(p3)
    p(f'Curvas sem suavização. Inicial: {ids("sac_hopper")}. Comparação: {ids(competitor)}.', "CaptionSmall")
    story.append(PageBreak())
    p("Parte 4 - Temperatura fixa", "Title")
    p("<b>Previsão registrada antes do treino.</b> " + conclusions["alpha_prediction"])
    p("<b>Escolha do alfa.</b> " + conclusions["alpha_choice"])
    pic(p4a)
    p(f'Referência: {ids(final, 2)}. Alfa fixo: {ids("sac_hopper_fixed_alpha")}.', "CaptionSmall")
    p("<b>Resultado e mecanismo.</b> " + conclusions["alpha_result"])
    p(f'Avaliação determinística: referência {stats(final, METRIC_EVAL, 2)} (seeds 1 e 2); '
      f'alfa fixo {stats("sac_hopper_fixed_alpha", METRIC_EVAL)} (2 seeds). '
      "As curvas sobrepõem apenas seeds 1 e 2 de cada configuração.")
    p("<b>Leitura conjunta.</b> " + conclusions["alpha_joint"])
    p("A entropia de uma densidade contínua pode ser negativa. Aqui, o alvo -3 corresponde "
      "às três dimensões da ação; a comparação usa a log-probabilidade corrigida pelo tanh.", "CaptionSmall")
    p("Referência técnica: " + conclusions["alpha_lines"])
    story.append(PageBreak())
    p("Parte 4 - Um crítico no alvo e ator", "Title")
    p("<b>Previsão registrada antes do treino.</b> " + conclusions["q_prediction"])
    pic(p4b)
    p(f'Referência: {ids(final, 2)}. Q1: {ids("sac_hopper_single_q")}.', "CaptionSmall")
    p("<b>Resultado e mecanismo.</b> " + conclusions["q_result"])
    p("Nesta implementação, clipped_double_q=false substitui o mínimo por Q1 no alvo e no ator. "
      "Q2 continua sendo calculado e treinado; a ablação testa o uso do mínimo, não a economia "
      "computacional de remover uma rede.")
    p(f'Avaliação determinística: configuração com Q1 {stats("sac_hopper_single_q", METRIC_EVAL)}. '
      "Os valores Q incluem desconto e entropia; não devem ser comparados diretamente com o retorno bruto.")
    p("Referência técnica: " + conclusions["q_lines"])
    p("<b>Limites da interpretação.</b> O diagnóstico Q - retorno usa um episódio observado, cuja política e "
      "temperatura podem mudar durante a coleta. No limite temporal, também ignora o bootstrap após a truncagem. "
      "No Hopper, gamma elevado a 1.000 é aproximadamente 0,000043. Dois seeds descrevem estes runs, "
      "mas não sustentam uma afirmação universal sobre a ablação.")
    story.append(PageBreak())
    p("Parte 5 - SAC versus PPO", "Title")
    p("<b>Previsão.</b> " + conclusions["ppo_prediction"])
    pic(p5a)
    pic(p5b)
    p(f'SAC: {ids(final, 2)}. PPO: {ids("ppo_pendulum")}. Tempo: campo _runtime do W&amp;B, em minutos; '
      "inclui a inicialização. Há duas lacunas de cerca de 34,5 e 14,7 min nos runs SAC; "
      "as linhas são interrompidas nas lacunas. O painel direito amplia a janela comum antes delas.",
      "CaptionSmall")
    p(conclusions["ppo_result"])
    p("A avaliação final original do PPO amostra ações. Para comparar desempenho ao longo do treino, "
      "as figuras usam a mesma métrica de retorno episódico nos dois algoritmos.", "CaptionSmall")
    p("<b>Reutilização das amostras.</b> " + conclusions["reuse"])
    p("<b>Orçamento efetivo.</b> O comando do enunciado configura 100.000 passos nos dois algoritmos. "
      "O PPO original fecha 24 rollouts de 4.096 transições: 98.304 passos, 1,696% abaixo do teto. "
      "O código PPO e seus hiperparâmetros foram preservados. O SAC conclui os 100.000 passos.")
    p("<b>Limite causal.</b> Replay, entropia, dois críticos e ator reparametrizado são mecanismos plausíveis, "
      "mas SAC e PPO também diferem nas redes e na normalização; a comparação não isola cada causa.")

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(46, 25, "SAC | Hopper-v5 | resultados reproduzíveis no W&B")
        canvas.drawRightString(A4[0] - 46, 25, str(doc.page))

    path = OUT / "Relatorio_SAC.pdf"
    SimpleDocTemplate(str(path), pagesize=A4, rightMargin=46, leftMargin=46,
                      topMargin=32, bottomMargin=40).build(story, onFirstPage=footer, onLaterPages=footer)
    from pypdf import PdfReader
    count = len(PdfReader(path).pages)
    if count > 4:
        raise ValueError(f"Report exceeds page limit: {count}")
    print(path)


METRIC_EVAL = "eval/mean_return"
METRIC_TRAIN = "charts/episodic_return_mean_last100"

if __name__ == "__main__":
    main()
