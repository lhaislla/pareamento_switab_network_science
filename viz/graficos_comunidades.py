"""Construcao de figuras Plotly para redes de comunidades e clusters."""

import math
from typing import List, Set

import networkx as nx
import plotly.graph_objects as go


def criar_figura_grafo(
    grafo: nx.DiGraph,
    titulo: str,
    comunidades: List[Set[str]],
    nome_algoritmo: str,
    exibir_setas: bool = True,
) -> go.Figure:
    """Cria figura Plotly colorindo nos por comunidade detectada e exibindo arestas direcionadas."""
    if grafo.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text=titulo, font=dict(size=18, color="#1F2937")),
            template="plotly_white",
            height=540,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    pos = nx.spring_layout(grafo, seed=42, k=0.45)
    edge_x, edge_y = [], []
    setas = []

    for origem, destino in grafo.edges():
        x0, y0 = pos[origem]
        x1, y1 = pos[destino]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        if exibir_setas and grafo.number_of_edges() <= 300:
            setas.append({
                "x": x1, "y": y1,
                "ax": x0, "ay": y0,
                "xref": "x", "yref": "y",
                "axref": "x", "ayref": "y",
                "showarrow": True,
                "arrowhead": 2,
                "arrowsize": 0.9,
                "arrowwidth": 1,
                "arrowcolor": "#B0B5BA",
            })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color="#B0B5BA"),
        hoverinfo="skip",
    ))

    nos = list(grafo.nodes())
    mapa_comunidade = {}
    for idx, com in enumerate(comunidades):
        for no in com:
            mapa_comunidade[no] = idx

    cores = [mapa_comunidade.get(no, -1) for no in nos]
    tamanhos = [12 + 3.5 * math.sqrt(grafo.degree(no)) for no in nos]
    customdata = [
        [
            str(no),
            grafo.nodes[no].get("nome", "N/A"),
            grafo.nodes[no].get("instituicao", "N/A"),
            grafo.nodes[no].get("turno", "N/A"),
            grafo.nodes[no].get("semestre", "N/A"),
            mapa_comunidade.get(no, "N/A"),
            grafo.in_degree(no),
            grafo.out_degree(no),
        ]
        for no in nos
    ]

    fig.add_trace(go.Scatter(
        x=[pos[no][0] for no in nos],
        y=[pos[no][1] for no in nos],
        mode="markers",
        marker=dict(
            size=tamanhos,
            color=cores,
            colorscale="Turbo",
            showscale=True,
            colorbar=dict(title=f"Comunidade<br>({nome_algoritmo})"),
            line=dict(width=1.2, color="white"),
        ),
        customdata=customdata,
        hovertemplate=(
            "<b>%{customdata[0]} - %{customdata[1]}</b><br>"
            "Instituição: %{customdata[2]} | Turno: %{customdata[3]}<br>"
            "Período: %{customdata[4]}<br>"
            "Comunidade: <b>%{customdata[5]}</b><br>"
            "Conexões recebidas (In-Degree): %{customdata[6]}<br>"
            "Conexões feitas (Out-Degree): %{customdata[7]}"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(text=titulo, font=dict(size=18, color="#1F2937")),
        annotations=setas,
        hovermode="closest",
        showlegend=False,
        template="plotly_white",
        height=580,
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
    )
    return fig
