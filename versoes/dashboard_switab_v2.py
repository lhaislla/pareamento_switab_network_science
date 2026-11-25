"""
Dashboard Interativo de Análise de Redes e Comunidades - Projeto SWITAB (v2).

Executa uma aplicação web local utilizando Plotly Dash com filtros dinâmicos
por instituição e seleção de algoritmo de detecção de comunidades (Louvain,
Label Propagation, Greedy Modularity / Clauset-Newman-Moore ou Girvan-Newman),
exibindo grafo interativo e painel de métricas topológicas.
"""

import math
from typing import Dict, List, Set, Tuple
import dash
from dash import dcc, html, Input, Output
import networkx as nx
import pandas as pd
import plotly.graph_objects as go


MAPA_FREQUENCIA: Dict[str, int] = {
    "Rara (apenas em período de avaliações)": 1,
    "Ocasional (algumas vezes no mês)": 2,
    "Frequente (semanalmente)": 3,
    "Muito frequente (diariamente)": 4,
}

ALGORITMOS_DISPONIVEIS: List[Dict[str, str]] = [
    {"label": "Louvain (Otimização de Modularidade Ponderada)", "value": "louvain"},
    {"label": "Label Propagation (Propagação de Rótulos)", "value": "label_propagation"},
    {"label": "Greedy Modularity (Clauset-Newman-Moore)", "value": "greedy_modularity"},
    {"label": "Girvan-Newman (Divisão por Intermediação / Betweenness)", "value": "girvan_newman"},
]


def carregar_dados(caminho_csv: str) -> pd.DataFrame:
    """Carrega a base de dados de conexões a partir de um arquivo CSV."""
    return pd.read_csv(caminho_csv)


def calcular_peso_vinculo(row: pd.Series) -> float:
    """Calcula o peso numérico da aresta com base na confiança técnica, conforto e frequência."""
    confianca = float(row.get("confianca_tecnica_1_a_5", 1.0))
    conforto = float(row.get("conforto_compartilhar_1_a_5", 1.0))
    freq_str = str(row.get("frequencia_interacao", ""))
    score_freq = float(MAPA_FREQUENCIA.get(freq_str, 1))
    return (confianca + conforto + score_freq) / 3.0


def construir_grafo_completo(df: pd.DataFrame) -> nx.DiGraph:
    """Constrói o grafo direcionado ponderado completo a partir do DataFrame."""
    grafo = nx.DiGraph()

    for _, row in df.iterrows():
        grafo.add_node(
            row["id_aluna"],
            nome=row.get("nome_completo", ""),
            instituicao=row.get("instituicao", ""),
            turno=row.get("turno", ""),
            semestre=row.get("semestre_atual", ""),
            participacao_projetos=row.get("participacao_projetos_p8", ""),
        )

    for _, row in df.iterrows():
        origem = row["id_aluna"]
        peso = calcular_peso_vinculo(row)
        conexoes_raw = row.get("colegas_conectadas_ids")

        if pd.notna(conexoes_raw) and str(conexoes_raw).strip():
            destinos = [d.strip() for d in str(conexoes_raw).split(";") if d.strip()]
            for destino in destinos:
                grafo.add_edge(origem, destino, weight=peso)

    return grafo


def extrair_subgrafo_institucional(
    grafo: nx.DiGraph, df: pd.DataFrame, instituicao: str
) -> nx.DiGraph:
    """Filtra o grafo completo para obter o subgrafo de uma instituição específica ou global."""
    if instituicao == "TODAS":
        return grafo.copy()
    ids_inst = df[df["instituicao"] == instituicao]["id_aluna"].tolist()
    return grafo.subgraph(ids_inst).copy()


def _converter_para_nao_direcionado_ponderado(grafo: nx.DiGraph) -> nx.Graph:
    """Converte um grafo direcionado em grafo simples não-direcionado somando pesos recíprocos."""
    grafo_nao_dir = nx.Graph()
    grafo_nao_dir.add_nodes_from(grafo.nodes(data=True))

    for u, v, data in grafo.edges(data=True):
        peso = float(data.get("weight", 1.0))
        if grafo_nao_dir.has_edge(u, v):
            grafo_nao_dir[u][v]["weight"] += peso
        else:
            grafo_nao_dir.add_edge(u, v, weight=peso)

    return grafo_nao_dir


def _selecionar_particao_girvan_newman(
    grafo_nao_dir: nx.Graph, max_comunidades: int = 6
) -> List[Set[str]]:
    """
    Executa Girvan-Newman e seleciona a partição com maior modularidade
    dentro de um limite razoável de comunidades para evitar fragmentação excessiva.
    """
    comp_generator = nx.community.girvan_newman(grafo_nao_dir)
    melhor_particao: Optional[List[Set[str]]] = None
    melhor_mod = -1.0

    for particao in comp_generator:
        comunidades_atuais = [set(c) for c in particao]
        if len(comunidades_atuais) > max_comunidades:
            break
        mod_atual = nx.community.modularity(
            grafo_nao_dir, comunidades_atuais, weight="weight"
        )
        if melhor_particao is None or mod_atual > melhor_mod:
            melhor_mod = mod_atual
            melhor_particao = comunidades_atuais

    if melhor_particao is None:
        return [set(grafo_nao_dir.nodes())]
    return melhor_particao


def detectar_comunidades(
    grafo: nx.DiGraph, algoritmo: str = "louvain", seed: int = 42
) -> Tuple[List[Set[str]], float, str]:
    """
    Executa o algoritmo de detecção de comunidades selecionado sobre a rede.

    Args:
        grafo: Grafo ou subgrafo direcionado ponderado.
        algoritmo: Identificador do algoritmo ('louvain', 'label_propagation',
            'greedy_modularity', 'girvan_newman').
        seed: Semente para reprodutibilidade quando aplicável.

    Returns:
        Tupla contendo (lista de comunidades, modularidade Q, nome legível do algoritmo).
    """
    grafo_nao_dir = _converter_para_nao_direcionado_ponderado(grafo)

    if grafo_nao_dir.number_of_edges() == 0:
        comunidades_isoladas = [{n} for n in grafo_nao_dir.nodes()]
        return comunidades_isoladas, 0.0, algoritmo

    if algoritmo == "louvain":
        comunidades_raw = nx.community.louvain_communities(
            grafo_nao_dir, weight="weight", seed=seed
        )
        comunidades = [set(c) for c in comunidades_raw]
        nome_alg = "Louvain"

    elif algoritmo == "label_propagation":
        comunidades_raw = nx.community.label_propagation_communities(grafo_nao_dir)
        comunidades = [set(c) for c in comunidades_raw]
        nome_alg = "Label Propagation"

    elif algoritmo == "greedy_modularity":
        comunidades_raw = nx.community.greedy_modularity_communities(
            grafo_nao_dir, weight="weight"
        )
        comunidades = [set(c) for c in comunidades_raw]
        nome_alg = "Greedy Modularity"

    elif algoritmo == "girvan_newman":
        comunidades = _selecionar_particao_girvan_newman(grafo_nao_dir)
        nome_alg = "Girvan-Newman"

    else:
        comunidades_raw = nx.community.louvain_communities(
            grafo_nao_dir, weight="weight", seed=seed
        )
        comunidades = [set(c) for c in comunidades_raw]
        nome_alg = "Louvain"

    modularidade = nx.community.modularity(
        grafo_nao_dir, comunidades, weight="weight"
    )
    return comunidades, modularidade, nome_alg


def calcular_metricas_rede(
    grafo: nx.DiGraph, comunidades: List[Set[str]], modularidade: float
) -> Dict[str, object]:
    """Calcula métricas topológicas e identifica o principal hub da rede filtrada."""
    graus_entrada = dict(grafo.in_degree())
    principal_hub = max(graus_entrada, key=graus_entrada.get) if graus_entrada else None
    nome_hub = (
        grafo.nodes[principal_hub].get("nome", "N/A") if principal_hub else "N/A"
    )

    grau_medio = (
        sum(dict(grafo.degree()).values()) / grafo.number_of_nodes()
        if grafo.number_of_nodes() > 0
        else 0.0
    )

    return {
        "num_nos": grafo.number_of_nodes(),
        "num_arestas": grafo.number_of_edges(),
        "densidade": round(nx.density(grafo), 4),
        "grau_medio": round(grau_medio, 2),
        "num_comunidades": len(comunidades),
        "modularidade": round(modularidade, 4),
        "principal_hub": f"{principal_hub} ({nome_hub})" if principal_hub else "N/A",
        "grau_entrada_max": graus_entrada.get(principal_hub, 0) if principal_hub else 0,
    }


def criar_figura_grafo(
    grafo: nx.DiGraph,
    titulo: str,
    comunidades: List[Set[str]],
    nome_algoritmo: str,
    exibir_setas: bool = True,
) -> go.Figure:
    """Gera a figura Plotly do grafo colorido pelas comunidades detectadas."""
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
        height=600,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
    )
    return fig


def criar_card_metrica(titulo: str, valor: object, subtitulo: str = "") -> html.Div:
    """Cria um componente visual de cartão (KPI Card) para exibir uma métrica."""
    return html.Div(
        style={
            "backgroundColor": "#FFFFFF",
            "border": "1px solid #E5E7EB",
            "borderRadius": "8px",
            "padding": "16px",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
            "flex": "1",
            "minWidth": "170px",
        },
        children=[
            html.Div(
                titulo,
                style={"fontSize": "12px", "color": "#6B7280", "fontWeight": "600", "textTransform": "uppercase"},
            ),
            html.Div(
                str(valor),
                style={"fontSize": "24px", "color": "#111827", "fontWeight": "700", "marginTop": "4px"},
            ),
            html.Div(
                subtitulo,
                style={"fontSize": "11px", "color": "#4B5563", "marginTop": "4px"},
            ) if subtitulo else None,
        ],
    )


# --- Inicialização dos Dados e Aplicação Dash ---
CAMINHO_CSV = r"data/base_conexoes_switab_ficticia.csv"
df_dados = carregar_dados(CAMINHO_CSV)
grafo_global = construir_grafo_completo(df_dados)

instituicoes_disponiveis = sorted(df_dados["instituicao"].dropna().unique().tolist())
opcoes_dropdown = [{"label": "Todas as Instituições (Rede Global)", "value": "TODAS"}] + [
    {"label": inst, "value": inst} for inst in instituicoes_disponiveis
]

app = dash.Dash(__name__, title="SWITAB - Análise de Comunidades")

app.layout = html.Div(
    style={"fontFamily": "Segoe UI, Arial, sans-serif", "backgroundColor": "#F9FAFB", "padding": "24px", "minHeight": "100vh"},
    children=[
        # Cabeçalho
        html.Div(
            style={"marginBottom": "20px"},
            children=[
                html.H1("SWITAB — Mapeamento de Conexões e Comunidades", style={"color": "#111827", "margin": "0 0 6px 0"}),
                html.P("Selecione a instituição e o algoritmo de detecção de comunidades para comparar partições e métricas.", style={"color": "#4B5563", "margin": "0"}),
            ],
        ),

        # Barra de Filtros
        html.Div(
            style={
                "backgroundColor": "#FFFFFF",
                "padding": "16px 20px",
                "borderRadius": "8px",
                "border": "1px solid #E5E7EB",
                "marginBottom": "20px",
                "display": "flex",
                "gap": "24px",
                "alignItems": "center",
                "flexWrap": "wrap",
            },
            children=[
                html.Div(
                    style={"minWidth": "240px", "flex": "1"},
                    children=[
                        html.Label("Instituição:", style={"fontWeight": "600", "fontSize": "14px", "color": "#374151", "display": "block", "marginBottom": "6px"}),
                        dcc.Dropdown(
                            id="filtro-instituicao",
                            options=opcoes_dropdown,
                            value="UPE",
                            clearable=False,
                        ),
                    ],
                ),
                html.Div(
                    style={"minWidth": "280px", "flex": "1"},
                    children=[
                        html.Label("Algoritmo de Comunidades:", style={"fontWeight": "600", "fontSize": "14px", "color": "#374151", "display": "block", "marginBottom": "6px"}),
                        dcc.Dropdown(
                            id="filtro-algoritmo",
                            options=ALGORITMOS_DISPONIVEIS,
                            value="louvain",
                            clearable=False,
                        ),
                    ],
                ),
                html.Div(
                    children=[
                        html.Label("Direção das Arestas:", style={"fontWeight": "600", "fontSize": "14px", "color": "#374151", "display": "block", "marginBottom": "6px"}),
                        dcc.RadioItems(
                            id="filtro-setas",
                            options=[
                                {"label": " Exibir setas", "value": "SIM"},
                                {"label": " Ocultar setas", "value": "NAO"},
                            ],
                            value="SIM",
                            inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "16px", "fontSize": "14px"},
                        ),
                    ]
                ),
            ],
        ),

        # Área do Grafo Interativo
        html.Div(
            style={"backgroundColor": "#FFFFFF", "borderRadius": "8px", "border": "1px solid #E5E7EB", "padding": "10px", "marginBottom": "20px"},
            children=[dcc.Graph(id="grafico-rede")],
        ),

        # Painel de Métricas Abaixo do Grafo
        html.Div(
            children=[
                html.H3(id="titulo-metricas", style={"color": "#111827", "marginBottom": "12px"}),
                html.Div(
                    id="painel-metricas",
                    style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                ),
            ]
        ),
    ],
)


@app.callback(
    [
        Output("grafico-rede", "figure"),
        Output("titulo-metricas", "children"),
        Output("painel-metricas", "children"),
    ],
    [
        Input("filtro-instituicao", "value"),
        Input("filtro-algoritmo", "value"),
        Input("filtro-setas", "value"),
    ],
)
def atualizar_dashboard(
    instituicao_selecionada: str, algoritmo_selecionado: str, exibir_setas_val: str
):
    """Atualiza o grafo e os cartões de métricas sempre que instituição, algoritmo ou setas mudam."""
    subgrafo = extrair_subgrafo_institucional(grafo_global, df_dados, instituicao_selecionada)
    comunidades, modularidade, nome_alg = detectar_comunidades(
        subgrafo, algoritmo=algoritmo_selecionado
    )
    metricas = calcular_metricas_rede(subgrafo, comunidades, modularidade)

    base_titulo = (
        "Rede Global (Todas as Instituições)"
        if instituicao_selecionada == "TODAS"
        else f"Rede de Conexões — {instituicao_selecionada}"
    )
    titulo_grafico = f"{base_titulo} | Algoritmo: {nome_alg}"

    fig = criar_figura_grafo(
        subgrafo,
        titulo=titulo_grafico,
        comunidades=comunidades,
        nome_algoritmo=nome_alg,
        exibir_setas=(exibir_setas_val == "SIM"),
    )

    titulo_secao_metricas = f"Métricas Topológicas e de Comunidade ({nome_alg})"

    cards = [
        criar_card_metrica("Total de Alunas (Nós)", metricas["num_nos"]),
        criar_card_metrica("Conexões (Arestas)", metricas["num_arestas"]),
        criar_card_metrica("Densidade da Rede", metricas["densidade"]),
        criar_card_metrica("Grau Médio", metricas["grau_medio"]),
        criar_card_metrica(f"Comunidades ({nome_alg})", metricas["num_comunidades"]),
        criar_card_metrica("Modularidade (Q)", metricas["modularidade"], "Qualidade da partição"),
        criar_card_metrica(
            "Principal Hub (In-Degree)",
            f"{metricas['grau_entrada_max']} conexões",
            str(metricas["principal_hub"]),
        ),
    ]

    return fig, titulo_secao_metricas, cards


if __name__ == "__main__":
    app.run(debug=True, port=8050)
