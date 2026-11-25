"""
Pipeline de Análise de Redes Sociais Acadêmicas e Detecção de Comunidades - Projeto SWITAB.

Este módulo implementa a carga de dados, construção de grafos direcionados ponderados,
particionamento institucional, detecção de comunidades (Louvain) e visualização interativa.
"""

import math
from typing import Dict, List, Optional, Set, Tuple
import networkx as nx
import pandas as pd
import plotly.graph_objects as go


MAPA_FREQUENCIA: Dict[str, int] = {
    "Rara (apenas em período de avaliações)": 1,
    "Ocasional (algumas vezes no mês)": 2,
    "Frequente (semanalmente)": 3,
    "Muito frequente (diariamente)": 4,
}


def carregar_dados(caminho_csv: str) -> pd.DataFrame:
    """
    Carrega a base de dados de conexões a partir de um arquivo CSV.

    Args:
        caminho_csv: Caminho relativo ou absoluto para o arquivo CSV.

    Returns:
        DataFrame contendo os registros das participantes e suas conexões.
    """
    return pd.read_csv(caminho_csv)


def calcular_peso_vinculo(row: pd.Series) -> float:
    """
    Calcula o peso numérico da aresta com base na confiança técnica, conforto e frequência.

    Args:
        row: Linha do DataFrame referente a uma participante.

    Returns:
        Peso médio normalizado do vínculo.
    """
    confianca = float(row.get("confianca_tecnica_1_a_5", 1.0))
    conforto = float(row.get("conforto_compartilhar_1_a_5", 1.0))
    freq_str = str(row.get("frequencia_interacao", ""))
    score_freq = float(MAPA_FREQUENCIA.get(freq_str, 1))
    return (confianca + conforto + score_freq) / 3.0


def construir_grafo_completo(df: pd.DataFrame) -> nx.DiGraph:
    """
    Constrói o grafo direcionado ponderado completo a partir do DataFrame.

    Corrige o desmembramento da lista de adjacência (separada por '; ') e anexa
    os metadados demográficos e acadêmicos como atributos dos nós.

    Args:
        df: DataFrame carregado com as respostas do formulário.

    Returns:
        Grafo direcionado (nx.DiGraph) com nós, atributos e arestas ponderadas.
    """
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
    """
    Filtra o grafo completo para obter o subgrafo isolado de uma instituição específica.

    Args:
        grafo: Grafo direcionado global.
        df: DataFrame original para filtragem dos IDs da instituição.
        instituicao: Nome da instituição (ex: 'UPE', 'UFPE', 'UFRPE', 'UNICAP', 'Cesar School').

    Returns:
        Subgrafo direcionado contendo apenas os nós e arestas da instituição selecionada.
    """
    ids_inst = df[df["instituicao"] == instituicao]["id_aluna"].tolist()
    return grafo.subgraph(ids_inst).copy()


def detectar_comunidades_louvain(
    grafo: nx.DiGraph, seed: int = 42
) -> Tuple[List[Set[str]], float]:
    """
    Converte o grafo para não-direcionado (agregando pesos recíprocos) e executa
    o algoritmo de Louvain para detecção de comunidades.

    Args:
        grafo: Grafo ou subgrafo direcionado ponderado.
        seed: Semente aleatória para reprodutibilidade do algoritmo.

    Returns:
        Tupla contendo a lista de conjuntos de nós (comunidades) e o valor de modularidade (Q).
    """
    grafo_nao_dir = nx.Graph()
    grafo_nao_dir.add_nodes_from(grafo.nodes(data=True))

    for u, v, data in grafo.edges(data=True):
        peso = data.get("weight", 1.0)
        if grafo_nao_dir.has_edge(u, v):
            grafo_nao_dir[u][v]["weight"] += peso
        else:
            grafo_nao_dir.add_edge(u, v, weight=peso)

    if grafo_nao_dir.number_of_edges() == 0:
        return [{n} for n in grafo_nao_dir.nodes()], 0.0

    comunidades = nx.community.louvain_communities(
        grafo_nao_dir, weight="weight", seed=seed
    )
    modularidade = nx.community.modularity(
        grafo_nao_dir, comunidades, weight="weight"
    )
    return comunidades, modularidade


def calcular_metricas_rede(
    grafo: nx.DiGraph, comunidades: Optional[List[Set[str]]] = None, modularidade: float = 0.0
) -> Dict[str, object]:
    """
    Calcula métricas topológicas básicas e identifica os principais hubs de entrada.

    Args:
        grafo: Grafo ou subgrafo direcionado.
        comunidades: Lista opcional de comunidades detectadas.
        modularidade: Score de modularidade associado às comunidades.

    Returns:
        Dicionário com resumo quantitativo das métricas topológicas.
    """
    graus_entrada = dict(grafo.in_degree())
    principal_hub = max(graus_entrada, key=graus_entrada.get) if graus_entrada else None

    return {
        "num_nos": grafo.number_of_nodes(),
        "num_arestas": grafo.number_of_edges(),
        "densidade": round(nx.density(grafo), 4),
        "num_comunidades": len(comunidades) if comunidades is not None else 0,
        "modularidade": round(modularidade, 4),
        "principal_hub_id": principal_hub,
        "grau_entrada_max": graus_entrada.get(principal_hub, 0) if principal_hub else 0,
    }


def gerar_visualizacao_interativa(
    grafo: nx.DiGraph,
    titulo: str = "Rede de Conexões entre Alunas",
    comunidades: Optional[List[Set[str]]] = None,
    salvar_html: Optional[str] = None,
    abrir_navegador: bool = True,
) -> go.Figure:
    """
    Gera um grafo interativo usando Plotly com setas direcionais e metadados no hover.

    Args:
        grafo: Grafo direcionado a ser plotado.
        titulo: Título superior do gráfico.
        comunidades: Lista opcional de comunidades para colorir os nós por cluster.
        salvar_html: Caminho opcional para exportar o gráfico como arquivo HTML autônomo.
        abrir_navegador: Se True, abre o gráfico diretamente no navegador padrão (evita erro de nbformat).

    Returns:
        Objeto Figure do Plotly.
    """
    pos = nx.spring_layout(grafo, seed=42)

    edge_x, edge_y = [], []
    setas = []

    for origem, destino in grafo.edges():
        x0, y0 = pos[origem]
        x1, y1 = pos[destino]

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        setas.append({
            "x": x1, "y": y1,
            "ax": x0, "ay": y0,
            "xref": "x", "yref": "y",
            "axref": "x", "ayref": "y",
            "showarrow": True,
            "arrowhead": 2,
            "arrowsize": 1,
            "arrowwidth": 1,
            "arrowcolor": "#9AA0A6",
        })

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color="#9AA0A6"),
        hoverinfo="skip",
    ))

    nos = list(grafo.nodes())
    mapa_comunidade = {}
    if comunidades:
        for idx, com in enumerate(comunidades):
            for no in com:
                mapa_comunidade[no] = idx

    if comunidades:
        cores = [mapa_comunidade.get(no, -1) for no in nos]
        titulo_colorbar = "Comunidade<br>(Louvain)"
        colorscale = "Turbo"
    else:
        cores = [grafo.in_degree(no) for no in nos]
        titulo_colorbar = "Conexões<br>recebidas"
        colorscale = "Viridis"

    tamanhos = [10 + 3 * math.sqrt(grafo.degree(no)) for no in nos]

    customdata = [
        [
            str(no),
            grafo.nodes[no].get("nome", "N/A"),
            grafo.nodes[no].get("turno", "N/A"),
            grafo.in_degree(no),
            grafo.out_degree(no),
            mapa_comunidade.get(no, "N/A"),
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
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title=titulo_colorbar),
            line=dict(width=1, color="white"),
        ),
        customdata=customdata,
        hovertemplate=(
            "<b>ID: %{customdata[0]}</b> (%{customdata[1]})<br>"
            "Turno: %{customdata[2]}<br>"
            "Comunidade: %{customdata[5]}<br>"
            "Conexões recebidas: %{customdata[3]}<br>"
            "Conexões feitas: %{customdata[4]}"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=titulo,
        annotations=setas,
        hovermode="closest",
        showlegend=False,
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
    )

    if salvar_html:
        fig.write_html(salvar_html)

    if abrir_navegador:
        fig.show(renderer="browser")

    return fig


def executar_pipeline(
    caminho_csv: str = r"data/base_conexoes_switab_ficticia.csv",
    instituicao_visualizacao: str = "UPE",
    salvar_html: Optional[str] = "rede_switab_upe.html",
    abrir_navegador: bool = True,
) -> pd.DataFrame:
    """
    Orquestra todo o pipeline: carga, modelagem, detecção de comunidades por instituição
    e visualização da rede selecionada.

    Args:
        caminho_csv: Caminho para o CSV de entrada.
        instituicao_visualizacao: Instituição que terá seu grafo renderizado.
        salvar_html: Caminho do arquivo HTML de saída para o gráfico interativo.
        abrir_navegador: Indica se deve abrir o gráfico no navegador.

    Returns:
        DataFrame consolidado com as métricas topológicas de todas as instituições.
    """
    df = carregar_dados(caminho_csv)
    grafo_global = construir_grafo_completo(df)

    resultados_instituicoes = []
    instituicoes = df["instituicao"].dropna().unique()

    for inst in instituicoes:
        subgrafo = extrair_subgrafo_institucional(grafo_global, df, inst)
        comunidades, modularidade = detectar_comunidades_louvain(subgrafo)
        metricas = calcular_metricas_rede(subgrafo, comunidades, modularidade)
        metricas["instituicao"] = inst
        resultados_instituicoes.append(metricas)

        if inst == instituicao_visualizacao:
            gerar_visualizacao_interativa(
                subgrafo,
                titulo=f"Rede de Conexões e Comunidades SWITAB - {inst}",
                comunidades=comunidades,
                salvar_html=salvar_html,
                abrir_navegador=abrir_navegador,
            )

    df_resultados = pd.DataFrame(resultados_instituicoes)
    colunas_ordenadas = [
        "instituicao",
        "num_nos",
        "num_arestas",
        "densidade",
        "num_comunidades",
        "modularidade",
        "principal_hub_id",
        "grau_entrada_max",
    ]
    return df_resultados[colunas_ordenadas]


if __name__ == "__main__":
    # Execução padrão do pipeline
    resumo_metricas = executar_pipeline(
        caminho_csv=r"data/base_conexoes_switab_ficticia.csv",
        instituicao_visualizacao="UPE",
        salvar_html="rede_switab_upe.html",
        abrir_navegador=True,
    )
    print("\nResumo Topológico por Instituição:")
    print(resumo_metricas.to_string(index=False))
