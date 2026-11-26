"""Construcao de figuras Plotly para visualizacao de pares de mentoria."""

import hashlib
from typing import Dict, Tuple

import networkx as nx
import plotly.graph_objects as go

_CACHE_POSICOES_GRAFO: Dict[str, Dict[str, Tuple[float, float]]] = {}


def _assinatura_grafo_layout(G: nx.Graph, tipo_grafico: str) -> str:
    """Gera assinatura deterministica da estrutura do grafo e tipo de layout."""
    nos = sorted(
        (n, round(float(d.get("ano_ingresso", 0.0)), 2), str(d.get("instituicao", "")))
        for n, d in G.nodes(data=True)
    )
    arestas = sorted(
        (u, v, round(float(d.get("afinidade", 0.0)), 4))
        for u, v, d in G.edges(data=True)
    )
    payload = f"{tipo_grafico}|{nos}|{arestas}".encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def _obter_posicoes_grafo(G: nx.Graph, tipo_grafico: str) -> Dict[str, Tuple[float, float]]:
    """Calcula ou recupera do cache as coordenadas (x, y) dos nos do grafo."""
    chave = _assinatura_grafo_layout(G, tipo_grafico)
    if chave in _CACHE_POSICOES_GRAFO:
        return _CACHE_POSICOES_GRAFO[chave]

    if tipo_grafico == "grafo_bipartido":
        u_nodes = sorted([n for n, d in G.nodes(data=True) if d["grupo_bipartido"] == "U"])
        v_nodes = sorted([n for n, d in G.nodes(data=True) if d["grupo_bipartido"] == "V"])
        pos: Dict[str, Tuple[float, float]] = {}
        for idx, n in enumerate(u_nodes):
            y = 1.0 - (idx / max(1, len(u_nodes) - 1)) if len(u_nodes) > 1 else 0.5
            pos[n] = (0.0, y)
        for idx, n in enumerate(v_nodes):
            y = 1.0 - (idx / max(1, len(v_nodes) - 1)) if len(v_nodes) > 1 else 0.5
            pos[n] = (1.0, y)
    else:
        G_sim = nx.Graph()
        for n, d in G.nodes(data=True):
            G_sim.add_node(n, **d)
        for u, v, d in G.edges(data=True):
            peso = max(0.05, float(d["afinidade"])) * 3.5
            G_sim.add_edge(u, v, weight=peso)
        pos = nx.spring_layout(G_sim, weight="weight", k=1.2, iterations=120, seed=42)

    if len(_CACHE_POSICOES_GRAFO) >= 128:
        _CACHE_POSICOES_GRAFO.clear()
    _CACHE_POSICOES_GRAFO[chave] = pos
    return pos


def gerar_grafico_similaridade_pares(
    G: nx.Graph,
    tipo_grafico: str = "similaridade_espacial",
    max_par: int = 2,
) -> go.Figure:
    """Gera figura Plotly para visualizacao por similaridade espacial ou grafo bipartido."""
    n_nodes = G.number_of_nodes()
    if n_nodes == 0:
        fig = go.Figure()
        fig.update_layout(
            title=dict(
                text="Pares Conectados — Sem pares para exibir",
                x=0.02,
                y=0.95,
                xanchor="left",
                yanchor="top",
                font=dict(size=15, color="#4a148c"),
            ),
            template="plotly_white",
            height=540,
            margin=dict(l=20, r=20, t=90, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        return fig

    pos = _obter_posicoes_grafo(G, tipo_grafico)

    ex, ey = [], []
    mid_x, mid_y, mid_hover = [], [], []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ex.extend([x0, x1, None])
        ey.extend([y0, y1, None])
        mid_x.append((x0 + x1) / 2.0)
        mid_y.append((y0 + y1) / 2.0)
        mid_hover.append(
            f"<b>Par: {u} ↔ {v}</b><br>"
            f"Afinidade: {d['afinidade']:.4f}<br>"
            f"Status: {d['status']}"
        )

    trace_arestas = go.Scatter(
        x=ex,
        y=ey,
        mode="lines",
        name="Pares / Conexões",
        line=dict(width=1.1, color="#90a4ae"),
        hoverinfo="none",
        opacity=0.65,
    )

    trace_mid_pares = go.Scatter(
        x=mid_x,
        y=mid_y,
        mode="markers",
        name="Detalhes do Par",
        hoverinfo="text",
        text=mid_hover,
        marker=dict(size=7, color="#37474f", opacity=0.0),
    )

    nos_mentoras = [n for n, d in G.nodes(data=True) if d["grupo_bipartido"] == "U"]
    nos_mentoradas = [n for n, d in G.nodes(data=True) if d["grupo_bipartido"] == "V" and d["papel"].lower() == "mentorada"]
    nos_demais_v = [n for n, d in G.nodes(data=True) if d["grupo_bipartido"] == "V" and d["papel"].lower() != "mentorada"]

    def montar_trace_papel(lista_nos: list, nome_legenda: str, cor_no: str, simbolo_no: str, tamanho_base: int) -> go.Scatter:
        x_vals = [pos[n][0] for n in lista_nos]
        y_vals = [pos[n][1] for n in lista_nos]
        hover_vals = []
        size_vals = []
        for n in lista_nos:
            d = G.nodes[n]
            deg = G.degree(n)
            size_vals.append(tamanho_base + min(6, deg * 1.5))
            hover_vals.append(
                f"<b>{n}</b> — {d['papel']}<br>"
                f"Instituição: {d['instituicao']}<br>"
                f"Curso: {d['curso']}<br>"
                f"Ingresso: {d['ano_ingresso']}<br>"
                f"Conexões (grau): {deg} (limite: {max_par})<br>"
                f"Áreas: {d['areas']}"
            )
        return go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            name=nome_legenda,
            hoverinfo="text",
            text=hover_vals,
            marker=dict(
                size=size_vals if len(size_vals) > 0 else tamanho_base,
                color=cor_no,
                symbol=simbolo_no,
                line=dict(width=1.3, color="#ffffff"),
                opacity=0.92,
            ),
        )

    traces_nos = [
        montar_trace_papel(nos_mentoras, "Mentora (U)", "#6a1b9a", "circle", 12),
        montar_trace_papel(nos_mentoradas, "Mentorada (V)", "#00838f", "circle", 10),
    ]
    if nos_demais_v:
        traces_nos.append(montar_trace_papel(nos_demais_v, "Embaixadora / Outro (V)", "#ef6c00", "diamond", 11))

    fig = go.Figure(data=[trace_arestas, trace_mid_pares] + traces_nos)
    subtitulo = (
        "Quanto mais próximos os nós, maior a similaridade/afinidade e conexão entre pares"
        if tipo_grafico == "similaridade_espacial"
        else "Visualização bipartida dos pares formados"
    )
    fig.update_layout(
        title=dict(
            text=f"Pares Conectados (Limite: {max_par} conexões/par)<br><span style='font-size:12px;color:#546e7a'>{subtitulo}</span>",
            x=0.02,
            y=0.97,
            xanchor="left",
            yanchor="top",
            font=dict(size=15, color="#4a148c"),
        ),
        template="plotly_white",
        height=580,
        margin=dict(l=20, r=20, t=125, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, domain=[0.0, 1.0]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, domain=[0.0, 0.92]),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.08,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0.85)",
        ),
    )
    return fig
