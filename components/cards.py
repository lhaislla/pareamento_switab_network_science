"""Componentes de cards de metricas e bloco de sintese textual da rede."""

from typing import Any, List, Optional

import dash_bootstrap_components as dbc
from dash import html
import networkx as nx
import numpy as np

from core.pareamento_engine import RelatorioIngestao


def criar_card_metrica(titulo: str, valor: object, subtitulo: str = "") -> html.Div:
    """Cria card visual padronizado para exibicao de metricas de rede."""
    return html.Div(
        style={
            "backgroundColor": "#FFFFFF",
            "border": "1px solid #E5E7EB",
            "borderRadius": "8px",
            "padding": "16px",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
            "flex": "1",
            "minWidth": "175px",
        },
        children=[
            html.Div(
                titulo,
                style={"fontSize": "12px", "color": "#6B7280", "fontWeight": "600", "textTransform": "uppercase"},
            ),
            html.Div(
                str(valor),
                style={"fontSize": "22px", "color": "#111827", "fontWeight": "700", "marginTop": "4px"},
            ),
            html.Div(
                subtitulo,
                style={"fontSize": "11px", "color": "#4B5563", "marginTop": "4px"},
            ) if subtitulo else None,
        ],
    )


def gerar_resumo_textual_complementar(
    G: nx.Graph,
    total_mentoradas_filtro: int,
    relatorio_ingestao: Optional[RelatorioIngestao] = None,
) -> html.Div:
    """Monta o bloco de sintese textual de cobertura, graus, afinidade e aderencia."""
    n_edges = G.number_of_edges()
    v_conectadas = sum(1 for _, d in G.nodes(data=True) if d["grupo_bipartido"] == "V")

    elementos: List[Any] = []

    if relatorio_ingestao and relatorio_ingestao.avisos:
        elementos.append(
            dbc.Alert(
                [
                    html.Strong("Auditoria da base: "),
                    " | ".join(relatorio_ingestao.avisos[:3]),
                ],
                color="info",
                style={"marginBottom": "8px", "fontSize": "0.80rem", "padding": "6px 10px"},
            )
        )

    if n_edges == 0:
        elementos.append(
            dbc.Alert(
                f"Nenhum par formado com os filtros atuais. Alunas conectadas: 0 de {total_mentoradas_filtro}.",
                color="warning",
                style={"marginBottom": "0", "fontSize": "0.88rem"},
            )
        )
        return html.Div(elementos)

    graus_m = [G.degree(n) for n, d in G.nodes(data=True) if d["grupo_bipartido"] == "U"]
    if graus_m:
        u_vals, u_cnt = np.unique(graus_m, return_counts=True)
        dist_grau_str = ", ".join([f"grau {k}: {c} mentora(s)" for k, c in zip(u_vals, u_cnt)])
    else:
        dist_grau_str = "sem mentoras conectadas"

    scores = [float(d["afinidade"]) for _, _, d in G.edges(data=True)]
    alta_cnt = sum(1 for s in scores if s >= 0.60)
    media_cnt = sum(1 for s in scores if 0.35 <= s < 0.60)
    baixa_cnt = sum(1 for s in scores if s < 0.35)
    afin_med = float(np.mean(scores))
    afin_min = float(np.min(scores))
    afin_max = float(np.max(scores))

    ader_per = 100.0 * float(np.mean([d["match_periodo"] for _, _, d in G.edges(data=True)]))
    ader_inst = 100.0 * float(np.mean([d["match_inst"] for _, _, d in G.edges(data=True)]))
    ader_curso = 100.0 * float(np.mean([d["match_curso"] for _, _, d in G.edges(data=True)]))

    pares_parciais_ou_fallback = sum(
        1
        for _, _, d in G.edges(data=True)
        if not (d["match_periodo"] and d["match_inst"] and d["match_curso"]) or d.get("excedeu_limite", False)
    )

    elementos.extend(
        [
            html.P(
                [
                    html.Strong("Cobertura das Alunas (V): "),
                    f"{v_conectadas} de {total_mentoradas_filtro} aluna(s) conectada(s) em {n_edges} par(es).",
                ],
                style={"fontSize": "0.86rem", "marginBottom": "6px"},
            ),
            html.P(
                [
                    html.Strong("Distribuição do Grau das Mentoras (U): "),
                    f"{dist_grau_str}.",
                ],
                style={"fontSize": "0.86rem", "marginBottom": "6px"},
            ),
            html.P(
                [
                    html.Strong("Escore de Afinidade por Par: "),
                    f"média {afin_med:.4f} (mín. {afin_min:.4f}, máx. {afin_max:.4f}) — "
                    f"alta (≥ 0,60): {alta_cnt} par(es), média (0,35 a 0,59): {media_cnt} par(es), baixa (< 0,35): {baixa_cnt} par(es).",
                ],
                style={"fontSize": "0.86rem", "marginBottom": "6px"},
            ),
            html.P(
                [
                    html.Strong("Aderência às Regras: "),
                    f"Período (>1 sem.): {ader_per:.1f}%, Instituição: {ader_inst:.1f}%, Curso: {ader_curso:.1f}%.",
                ],
                style={"fontSize": "0.86rem", "marginBottom": "6px"},
            ),
            html.P(
                [
                    html.Strong("Pares com Exceção / Flexibilização: "),
                    f"{pares_parciais_ou_fallback} de {n_edges} par(es) apresentam alguma regra não atendida ou limite excedido.",
                ],
                style={"fontSize": "0.84rem", "marginBottom": "0", "color": "#495057"},
            ),
        ]
    )

    return html.Div(elementos)
