"""
Dashboard Interativo de Análise de Redes e Comunidades - Projeto SWITAB (v4).

Executa uma aplicação web local utilizando Plotly Dash com:
1. Filtro de instituição (incluindo "Instituição Geral (Rede Completa)") e filtro por turno.
2. Filtro de algoritmo de detecção de comunidades (Louvain, Label Propagation,
   Greedy Modularity ou Girvan-Newman) e seleção do critério de peso das arestas.
3. Grafo interativo com destaque de comunidades e cartões de métricas (incluindo
   homofilia de turno, intermediação/betweenness e hub qualitativo).
4. Resumo detalhado por comunidade detectada.
5. Tabela comparativa completa exibindo as métricas de todos os algoritmos para
   todas as instituições (e rede geral), com botão de exportação em CSV.
"""

from functools import lru_cache
import io
import math
from typing import Dict, List, Optional, Set, Tuple
import dash
from dash import dash_table, dcc, html, Input, Output, State
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

CRITERIOS_PESO: List[Dict[str, str]] = [
    {"label": "Composto (Confiança + Conforto + Frequência)", "value": "composto"},
    {"label": "Apenas Confiança Técnica (1 a 5)", "value": "confianca"},
    {"label": "Apenas Conforto Emocional (1 a 5)", "value": "conforto"},
    {"label": "Apenas Frequência de Interação (1 a 4)", "value": "frequencia"},
]


def carregar_dados(caminho_csv: str) -> pd.DataFrame:
    """Carrega a base de dados de conexões a partir de um arquivo CSV."""
    return pd.read_csv(caminho_csv)


def calcular_peso_vinculo(row: pd.Series, criterio: str = "composto") -> float:
    """
    Calcula o peso numérico da aresta de acordo com o critério selecionado.

    Args:
        row: Linha do DataFrame referente a uma participante.
        criterio: 'composto', 'confianca', 'conforto' ou 'frequencia'.

    Returns:
        Peso numérico da aresta.
    """
    confianca = float(row.get("confianca_tecnica_1_a_5", 1.0))
    conforto = float(row.get("conforto_compartilhar_1_a_5", 1.0))
    freq_str = str(row.get("frequencia_interacao", ""))
    score_freq = float(MAPA_FREQUENCIA.get(freq_str, 1))

    if criterio == "confianca":
        return confianca
    if criterio == "conforto":
        return conforto
    if criterio == "frequencia":
        return score_freq
    return (confianca + conforto + score_freq) / 3.0


def construir_grafo_completo(df: pd.DataFrame, criterio_peso: str = "composto") -> nx.DiGraph:
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
            pessoa_mais_conectada=row.get("pessoa_mais_conectada", ""),
        )

    for _, row in df.iterrows():
        origem = row["id_aluna"]
        peso = calcular_peso_vinculo(row, criterio=criterio_peso)
        conexoes_raw = row.get("colegas_conectadas_ids")

        if pd.notna(conexoes_raw) and str(conexoes_raw).strip():
            destinos = [d.strip() for d in str(conexoes_raw).split(";") if d.strip()]
            for destino in destinos:
                grafo.add_edge(origem, destino, weight=peso)

    return grafo


def extrair_subgrafo_filtrado(
    grafo: nx.DiGraph,
    df: pd.DataFrame,
    instituicao: str,
    turno: str = "TODOS",
) -> nx.DiGraph:
    """Filtra o grafo por instituição (ou rede geral) e opcionalmente por turno."""
    df_filtrado = df.copy()
    if instituicao != "GERAL":
        df_filtrado = df_filtrado[df_filtrado["instituicao"] == instituicao]
    if turno != "TODOS":
        df_filtrado = df_filtrado[df_filtrado["turno"] == turno]

    ids_selecionados = df_filtrado["id_aluna"].tolist()
    return grafo.subgraph(ids_selecionados).copy()


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
    """Executa o algoritmo de detecção de comunidades selecionado sobre a rede."""
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


def calcular_homofilia_turno(grafo: nx.DiGraph) -> float:
    """Calcula o percentual de arestas que conectam alunas do mesmo turno."""
    if grafo.number_of_edges() == 0:
        return 0.0
    mesmo_turno = 0
    for u, v in grafo.edges():
        turno_u = grafo.nodes[u].get("turno", "")
        turno_v = grafo.nodes[v].get("turno", "")
        if turno_u and turno_u == turno_v:
            mesmo_turno += 1
    return round((mesmo_turno / grafo.number_of_edges()) * 100.0, 1)


def calcular_metricas_rede(
    grafo: nx.DiGraph, comunidades: List[Set[str]], modularidade: float
) -> Dict[str, object]:
    """Calcula métricas topológicas, centralidade de intermediação, homofilia e hubs."""
    graus_entrada = dict(grafo.in_degree())
    principal_hub = max(graus_entrada, key=graus_entrada.get) if graus_entrada else None
    nome_hub = (
        grafo.nodes[principal_hub].get("nome", "N/A") if principal_hub else "N/A"
    )

    betweenness = nx.betweenness_centrality(grafo, weight="weight") if grafo.number_of_nodes() > 0 else {}
    ponte_id = max(betweenness, key=betweenness.get) if betweenness else None
    nome_ponte = grafo.nodes[ponte_id].get("nome", "N/A") if ponte_id else "N/A"

    grau_medio = (
        sum(dict(grafo.degree()).values()) / grafo.number_of_nodes()
        if grafo.number_of_nodes() > 0
        else 0.0
    )

    homofilia_turno = calcular_homofilia_turno(grafo)

    return {
        "num_nos": grafo.number_of_nodes(),
        "num_arestas": grafo.number_of_edges(),
        "densidade": round(nx.density(grafo), 4),
        "grau_medio": round(grau_medio, 2),
        "num_comunidades": len(comunidades),
        "modularidade": round(modularidade, 4),
        "homofilia_turno": homofilia_turno,
        "principal_hub": f"{principal_hub} ({nome_hub})" if principal_hub else "N/A",
        "grau_entrada_max": graus_entrada.get(principal_hub, 0) if principal_hub else 0,
        "principal_ponte": f"{ponte_id} ({nome_ponte})" if ponte_id else "N/A",
        "betweenness_max": round(betweenness.get(ponte_id, 0.0), 3) if ponte_id else 0.0,
    }


def gerar_resumo_comunidades(
    grafo: nx.DiGraph, comunidades: List[Set[str]]
) -> List[Dict[str, object]]:
    """Gera uma tabela resumo com tamanho, turno predominante e líder por comunidade."""
    linhas: List[Dict[str, object]] = []
    for idx, membros in enumerate(comunidades):
        membros_lista = sorted(list(membros))
        turnos_membros = [
            grafo.nodes[n].get("turno", "N/A") for n in membros_lista if grafo.has_node(n)
        ]
        turno_predominante = (
            pd.Series(turnos_membros).mode().iloc[0] if turnos_membros else "N/A"
        )

        lider = max(membros_lista, key=lambda n: grafo.degree(n)) if membros_lista else "N/A"
        nome_lider = grafo.nodes[lider].get("nome", "") if grafo.has_node(lider) else ""

        linhas.append({
            "comunidade_id": f"Comunidade {idx}",
            "qtd_alunas": len(membros_lista),
            "turno_predominante": turno_predominante,
            "no_mais_conectado": f"{lider} ({nome_lider})" if nome_lider else lider,
        })

    linhas.sort(key=lambda x: x["qtd_alunas"], reverse=True)
    return linhas


@lru_cache(maxsize=64)
def _calcular_tabela_comparativa_cached(
    criterio_peso: str, filtro_instituicao: str
) -> Tuple[Dict[str, object], ...]:
    """Versão com cache em memória da tabela comparativa de todos os algoritmos."""
    df = carregar_dados(CAMINHO_CSV)
    grafo_global = construir_grafo_completo(df, criterio_peso=criterio_peso)

    escopos = ["GERAL"] + sorted(df["instituicao"].dropna().unique().tolist())
    if filtro_instituicao != "TODAS_LINHAS":
        escopos = [filtro_instituicao]

    linhas: List[Dict[str, object]] = []

    for escopo in escopos:
        rotulo_inst = "Instituição Geral (Rede Completa)" if escopo == "GERAL" else escopo
        subgrafo = extrair_subgrafo_filtrado(grafo_global, df, escopo, turno="TODOS")

        for alg_info in ALGORITMOS_DISPONIVEIS:
            alg_cod = alg_info["value"]
            comunidades, modularidade, nome_alg = detectar_comunidades(
                subgrafo, algoritmo=alg_cod
            )
            metricas = calcular_metricas_rede(subgrafo, comunidades, modularidade)

            linhas.append({
                "instituicao": rotulo_inst,
                "algoritmo": nome_alg,
                "num_nos": metricas["num_nos"],
                "num_arestas": metricas["num_arestas"],
                "densidade": metricas["densidade"],
                "grau_medio": metricas["grau_medio"],
                "homofilia_turno": metricas["homofilia_turno"],
                "num_comunidades": metricas["num_comunidades"],
                "modularidade": metricas["modularidade"],
                "principal_hub": metricas["principal_hub"],
            })

    return tuple(linhas)


def gerar_tabela_comparativa(
    criterio_peso: str = "composto", filtro_instituicao: str = "TODAS_LINHAS"
) -> List[Dict[str, object]]:
    """Retorna lista de dicionários para a DataTable comparativa."""
    return list(_calcular_tabela_comparativa_cached(criterio_peso, filtro_instituicao))


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


# --- Inicialização dos Dados e Aplicação Dash ---
CAMINHO_CSV = r"data/base_conexoes_switab_ficticia.csv"
df_dados = carregar_dados(CAMINHO_CSV)

instituicoes_disponiveis = sorted(df_dados["instituicao"].dropna().unique().tolist())
turnos_disponiveis = sorted(df_dados["turno"].dropna().unique().tolist())

opcoes_dropdown_inst = [
    {"label": "Instituição Geral (Rede Completa)", "value": "GERAL"}
] + [{"label": inst, "value": inst} for inst in instituicoes_disponiveis]

opcoes_dropdown_turno = [
    {"label": "Todos os Turnos", "value": "TODOS"}
] + [{"label": t, "value": t} for t in turnos_disponiveis]

opcoes_filtro_tabela = [
    {"label": "Todas as Instituições + Instituição Geral", "value": "TODAS_LINHAS"},
    {"label": "Apenas Instituição Geral (Rede Completa)", "value": "GERAL"},
] + [{"label": inst, "value": inst} for inst in instituicoes_disponiveis]

colunas_tabela_comparativa = [
    {"name": "Instituição", "id": "instituicao"},
    {"name": "Algoritmo", "id": "algoritmo"},
    {"name": "Nós", "id": "num_nos"},
    {"name": "Arestas", "id": "num_arestas"},
    {"name": "Densidade", "id": "densidade"},
    {"name": "Grau Médio", "id": "grau_medio"},
    {"name": "Homofilia Turno (%)", "id": "homofilia_turno"},
    {"name": "Comunidades", "id": "num_comunidades"},
    {"name": "Modularidade (Q)", "id": "modularidade"},
    {"name": "Principal Hub", "id": "principal_hub"},
]

colunas_tabela_comunidades = [
    {"name": "Comunidade", "id": "comunidade_id"},
    {"name": "Qtd. Alunas", "id": "qtd_alunas"},
    {"name": "Turno Predominante", "id": "turno_predominante"},
    {"name": "Nó Mais Conectado", "id": "no_mais_conectado"},
]

app = dash.Dash(__name__, title="SWITAB - Análise de Comunidades")

app.layout = html.Div(
    style={"fontFamily": "Segoe UI, Arial, sans-serif", "backgroundColor": "#F9FAFB", "padding": "24px", "minHeight": "100vh"},
    children=[
        html.Div(
            style={"marginBottom": "20px"},
            children=[
                html.H1("SWITAB — Mapeamento de Conexões e Comunidades", style={"color": "#111827", "margin": "0 0 6px 0"}),
                html.P("Explore por instituição ou rede geral, compare algoritmos de comunidades, inspecione clusters e exporte métricas.", style={"color": "#4B5563", "margin": "0"}),
            ],
        ),

        html.Div(
            style={
                "backgroundColor": "#FFFFFF",
                "padding": "16px 20px",
                "borderRadius": "8px",
                "border": "1px solid #E5E7EB",
                "marginBottom": "20px",
                "display": "flex",
                "gap": "20px",
                "alignItems": "center",
                "flexWrap": "wrap",
            },
            children=[
                html.Div(
                    style={"minWidth": "220px", "flex": "1"},
                    children=[
                        html.Label("Instituição / Escopo:", style={"fontWeight": "600", "fontSize": "13px", "color": "#374151", "display": "block", "marginBottom": "6px"}),
                        dcc.Dropdown(
                            id="filtro-instituicao",
                            options=opcoes_dropdown_inst,
                            value="GERAL",
                            clearable=False,
                        ),
                    ],
                ),
                html.Div(
                    style={"minWidth": "170px", "flex": "0.7"},
                    children=[
                        html.Label("Turno:", style={"fontWeight": "600", "fontSize": "13px", "color": "#374151", "display": "block", "marginBottom": "6px"}),
                        dcc.Dropdown(
                            id="filtro-turno",
                            options=opcoes_dropdown_turno,
                            value="TODOS",
                            clearable=False,
                        ),
                    ],
                ),
                html.Div(
                    style={"minWidth": "240px", "flex": "1"},
                    children=[
                        html.Label("Algoritmo de Comunidades:", style={"fontWeight": "600", "fontSize": "13px", "color": "#374151", "display": "block", "marginBottom": "6px"}),
                        dcc.Dropdown(
                            id="filtro-algoritmo",
                            options=ALGORITMOS_DISPONIVEIS,
                            value="louvain",
                            clearable=False,
                        ),
                    ],
                ),
                html.Div(
                    style={"minWidth": "220px", "flex": "1"},
                    children=[
                        html.Label("Peso da Aresta:", style={"fontWeight": "600", "fontSize": "13px", "color": "#374151", "display": "block", "marginBottom": "6px"}),
                        dcc.Dropdown(
                            id="filtro-peso",
                            options=CRITERIOS_PESO,
                            value="composto",
                            clearable=False,
                        ),
                    ],
                ),
                html.Div(
                    children=[
                        html.Label("Direção das Arestas:", style={"fontWeight": "600", "fontSize": "13px", "color": "#374151", "display": "block", "marginBottom": "6px"}),
                        dcc.RadioItems(
                            id="filtro-setas",
                            options=[
                                {"label": " Exibir setas", "value": "SIM"},
                                {"label": " Ocultar setas", "value": "NAO"},
                            ],
                            value="SIM",
                            inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "12px", "fontSize": "13px"},
                        ),
                    ]
                ),
            ],
        ),

        html.Div(
            style={"backgroundColor": "#FFFFFF", "borderRadius": "8px", "border": "1px solid #E5E7EB", "padding": "10px", "marginBottom": "20px"},
            children=[dcc.Graph(id="grafico-rede")],
        ),

        html.Div(
            style={"marginBottom": "28px"},
            children=[
                html.H3(id="titulo-metricas", style={"color": "#111827", "marginBottom": "12px"}),
                html.Div(
                    id="painel-metricas",
                    style={"display": "flex", "gap": "14px", "flexWrap": "wrap"},
                ),
            ],
        ),

        html.Div(
            style={
                "backgroundColor": "#FFFFFF",
                "padding": "20px",
                "borderRadius": "8px",
                "border": "1px solid #E5E7EB",
                "marginBottom": "28px",
            },
            children=[
                html.H3("Resumo das Comunidades Detectadas na Seleção Atual", style={"color": "#111827", "margin": "0 0 12px 0"}),
                dash_table.DataTable(
                    id="tabela-resumo-comunidades",
                    columns=colunas_tabela_comunidades,
                    data=[],
                    sort_action="native",
                    page_size=10,
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": "#374151",
                        "color": "#FFFFFF",
                        "fontWeight": "600",
                        "fontSize": "13px",
                        "textAlign": "left",
                        "padding": "9px",
                    },
                    style_cell={
                        "fontSize": "13px",
                        "padding": "8px 9px",
                        "borderBottom": "1px solid #E5E7EB",
                        "textAlign": "left",
                        "color": "#111827",
                    },
                ),
            ],
        ),

        html.Div(
            style={
                "backgroundColor": "#FFFFFF",
                "padding": "20px",
                "borderRadius": "8px",
                "border": "1px solid #E5E7EB",
            },
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "flexWrap": "wrap", "gap": "16px", "marginBottom": "16px"},
                    children=[
                        html.Div(
                            children=[
                                html.H3("Tabela Comparativa: Todos os Algoritmos por Instituição", style={"color": "#111827", "margin": "0 0 4px 0"}),
                                html.P("Compare comunidades, modularidade (Q), homofilia e hubs de cada algoritmo em cada instituição ou na rede geral.", style={"color": "#4B5563", "margin": "0", "fontSize": "13px"}),
                            ]
                        ),
                        html.Div(
                            style={"display": "flex", "gap": "12px", "alignItems": "flex-end", "flexWrap": "wrap"},
                            children=[
                                html.Div(
                                    style={"minWidth": "260px"},
                                    children=[
                                        html.Label("Filtrar Tabela por Instituição:", style={"fontWeight": "600", "fontSize": "13px", "color": "#374151", "display": "block", "marginBottom": "4px"}),
                                        dcc.Dropdown(
                                            id="filtro-tabela-instituicao",
                                            options=opcoes_filtro_tabela,
                                            value="TODAS_LINHAS",
                                            clearable=False,
                                        ),
                                    ],
                                ),
                                html.Button(
                                    "Exportar CSV",
                                    id="btn-exportar-csv",
                                    n_clicks=0,
                                    style={
                                        "backgroundColor": "#1F2937",
                                        "color": "#FFFFFF",
                                        "border": "none",
                                        "borderRadius": "6px",
                                        "padding": "9px 16px",
                                        "fontSize": "13px",
                                        "fontWeight": "600",
                                        "cursor": "pointer",
                                    },
                                ),
                                dcc.Download(id="download-tabela-csv"),
                            ],
                        ),
                    ],
                ),
                dash_table.DataTable(
                    id="tabela-comparativa-metricas",
                    columns=colunas_tabela_comparativa,
                    data=[],
                    sort_action="native",
                    filter_action="native",
                    page_size=24,
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": "#1F2937",
                        "color": "#FFFFFF",
                        "fontWeight": "600",
                        "fontSize": "13px",
                        "textAlign": "left",
                        "padding": "10px",
                    },
                    style_cell={
                        "fontSize": "13px",
                        "padding": "9px 10px",
                        "borderBottom": "1px solid #E5E7EB",
                        "textAlign": "left",
                        "color": "#111827",
                    },
                    style_data_conditional=[
                        {
                            "if": {"row_index": "odd"},
                            "backgroundColor": "#F9FAFB",
                        },
                        {
                            "if": {"filter_query": '{instituicao} = "Instituição Geral (Rede Completa)"'},
                            "fontWeight": "600",
                        },
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    [
        Output("grafico-rede", "figure"),
        Output("titulo-metricas", "children"),
        Output("painel-metricas", "children"),
        Output("tabela-resumo-comunidades", "data"),
    ],
    [
        Input("filtro-instituicao", "value"),
        Input("filtro-turno", "value"),
        Input("filtro-algoritmo", "value"),
        Input("filtro-peso", "value"),
        Input("filtro-setas", "value"),
    ],
)
def atualizar_grafo_e_cards(
    instituicao_selecionada: str,
    turno_selecionado: str,
    algoritmo_selecionado: str,
    criterio_peso: str,
    exibir_setas_val: str,
):
    """Atualiza o grafo, cartões de métricas e resumo por comunidade."""
    grafo_ponderado = construir_grafo_completo(df_dados, criterio_peso=criterio_peso)
    subgrafo = extrair_subgrafo_filtrado(
        grafo_ponderado, df_dados, instituicao_selecionada, turno=turno_selecionado
    )
    comunidades, modularidade, nome_alg = detectar_comunidades(
        subgrafo, algoritmo=algoritmo_selecionado
    )
    metricas = calcular_metricas_rede(subgrafo, comunidades, modularidade)
    resumo_comunidades = gerar_resumo_comunidades(subgrafo, comunidades)

    base_titulo = (
        "Instituição Geral (Rede Completa)"
        if instituicao_selecionada == "GERAL"
        else f"Rede de Conexões — {instituicao_selecionada}"
    )
    if turno_selecionado != "TODOS":
        base_titulo += f" ({turno_selecionado})"

    titulo_grafico = f"{base_titulo} | Algoritmo: {nome_alg}"

    fig = criar_figura_grafo(
        subgrafo,
        titulo=titulo_grafico,
        comunidades=comunidades,
        nome_algoritmo=nome_alg,
        exibir_setas=(exibir_setas_val == "SIM"),
    )

    titulo_secao_metricas = f"Métricas Topológicas e de Comunidade ({base_titulo} — {nome_alg})"

    cards = [
        criar_card_metrica("Total de Alunas (Nós)", metricas["num_nos"]),
        criar_card_metrica("Conexões (Arestas)", metricas["num_arestas"]),
        criar_card_metrica("Densidade da Rede", metricas["densidade"]),
        criar_card_metrica("Grau Médio", metricas["grau_medio"]),
        criar_card_metrica("Homofilia de Turno", f"{metricas['homofilia_turno']}%", "Mesmo turno na aresta"),
        criar_card_metrica(f"Comunidades ({nome_alg})", metricas["num_comunidades"]),
        criar_card_metrica("Modularidade (Q)", metricas["modularidade"], "Qualidade da partição"),
        criar_card_metrica(
            "Principal Hub (In-Degree)",
            f"{metricas['grau_entrada_max']} conexões",
            str(metricas["principal_hub"]),
        ),
        criar_card_metrica(
            "Maior Ponte (Betweenness)",
            str(metricas["betweenness_max"]),
            str(metricas["principal_ponte"]),
        ),
    ]

    return fig, titulo_secao_metricas, cards, resumo_comunidades


@app.callback(
    Output("tabela-comparativa-metricas", "data"),
    [
        Input("filtro-tabela-instituicao", "value"),
        Input("filtro-peso", "value"),
    ],
)
def atualizar_tabela_comparativa(filtro_tabela: str, criterio_peso: str):
    """Preenche a tabela comparativa com todos os algoritmos para as instituições."""
    return gerar_tabela_comparativa(
        criterio_peso=criterio_peso, filtro_instituicao=filtro_tabela
    )


@app.callback(
    Output("download-tabela-csv", "data"),
    [Input("btn-exportar-csv", "n_clicks")],
    [
        State("filtro-tabela-instituicao", "value"),
        State("filtro-peso", "value"),
    ],
    prevent_initial_call=True,
)
def exportar_tabela_csv(n_clicks: int, filtro_tabela: str, criterio_peso: str):
    """Exporta os dados atuais da tabela comparativa para download em CSV."""
    dados = gerar_tabela_comparativa(
        criterio_peso=criterio_peso, filtro_instituicao=filtro_tabela
    )
    df_export = pd.DataFrame(dados)
    buffer = io.StringIO()
    df_export.to_csv(buffer, index=False, encoding="utf-8-sig")
    return dict(content=buffer.getvalue(), filename="comparativo_algoritmos_switab.csv")


if __name__ == "__main__":
    app.run(debug=True, port=8050)