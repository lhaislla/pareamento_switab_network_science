"""Funcoes de construcao de grafo, deteccao de comunidades e metricas topologicas."""

import os
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

MAPA_FREQUENCIA: Dict[str, int] = {
    "Rara (apenas em período de avaliações)": 1,
    "Ocasional (algumas vezes no mês)": 2,
    "Frequente (semanalmente)": 3,
    "Muito frequente (diariamente)": 4,
}

ALGORITMOS_DISPONIVEIS: List[Dict[str, str]] = [
    {"label": "Louvain (Otimização de Modularidade Ponderada)", "value": "louvain"},
    {"label": "Label Propagation Assíncrona (Votação Ponderada)", "value": "label_propagation"},
    {"label": "Greedy Modularity Ponderada (Clauset-Newman-Moore)", "value": "greedy_modularity"},
    {"label": "Girvan-Newman (Betweenness com Distância Inversa)", "value": "girvan_newman"},
]

CRITERIOS_PESO: List[Dict[str, str]] = [
    {"label": "Composto (Confiança + Conforto + Frequência)", "value": "composto"},
    {"label": "Apenas Confiança Técnica (1 a 5)", "value": "confianca"},
    {"label": "Apenas Conforto Emocional (1 a 5)", "value": "conforto"},
    {"label": "Apenas Frequência de Interação (1 a 4)", "value": "frequencia"},
]


def gerar_dataset_comunidades_sintetico(seed: int = 42) -> pd.DataFrame:
    """Gera base sintetica de conexoes entre participantes para analise de comunidades."""
    rng = np.random.default_rng(seed)
    insts = ["UPE", "UFRPE", "UFPE", "UNICAP", "CESAR School"]
    turnos = ["Manhã", "Tarde", "Noite", "Integral"]
    semestres = ["2023.1", "2023.2", "2024.1", "2024.2", "2025.1", "2025.2", "2026.1"]
    freqs = list(MAPA_FREQUENCIA.keys())
    n_alunas = 45
    ids = [f"A{i+1:03d}" for i in range(n_alunas)]
    linhas = []
    for idx, aid in enumerate(ids):
        inst = insts[idx % len(insts)]
        turno = turnos[(idx + idx // len(insts)) % len(turnos)]
        sem = semestres[idx % len(semestres)]
        candidatos = [x for x in ids if x != aid]
        n_con = int(rng.integers(2, 6))
        con_escolhidas = sorted(rng.choice(candidatos, size=n_con, replace=False).tolist())
        linhas.append({
            "id_aluna": aid,
            "nome_completo": f"Participante {aid}",
            "instituicao": inst,
            "turno": turno,
            "semestre_atual": sem,
            "participacao_projetos_p8": "Sim" if idx % 3 == 0 else "Não",
            "pessoa_mais_conectada": con_escolhidas[0] if con_escolhidas else "",
            "colegas_conectadas_ids": ";".join(con_escolhidas),
            "confianca_tecnica_1_a_5": int(rng.integers(2, 6)),
            "conforto_compartilhar_1_a_5": int(rng.integers(2, 6)),
            "frequencia_interacao": str(rng.choice(freqs)),
        })
    return pd.DataFrame(linhas)


def carregar_dados_comunidades(caminho_csv: str) -> Tuple[pd.DataFrame, str]:
    """Carrega a base de comunidades do disco e falha se ela não puder ser usada."""
    if not caminho_csv:
        raise FileNotFoundError(f"Arquivo de comunidades não encontrado: {caminho_csv}")
    if not os.path.isabs(caminho_csv):
        raiz_projeto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        caminho_csv = os.path.join(raiz_projeto, caminho_csv)
    if not os.path.isfile(caminho_csv):
        raise FileNotFoundError(f"Arquivo de comunidades não encontrado: {caminho_csv}")

    df = pd.read_csv(caminho_csv)
    if df.empty:
        raise ValueError(f"Arquivo de comunidades vazio: {caminho_csv}")

    colunas_obrigatorias = {"id_aluna", "instituicao", "turno", "colegas_conectadas_ids"}
    colunas_ausentes = sorted(colunas_obrigatorias - set(df.columns))
    if colunas_ausentes:
        raise ValueError(
            f"Arquivo de comunidades sem colunas obrigatórias "
            f"({', '.join(colunas_ausentes)}): {caminho_csv}"
        )

    return df, f"Arquivo: {caminho_csv}"


def calcular_peso_vinculo(row: pd.Series, criterio: str = "composto") -> float:
    """Calcula peso numerico da aresta conforme criterio selecionado."""
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
    """Constroi grafo direcionado ponderado das conexoes entre participantes."""
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
                if grafo.has_node(destino):
                    grafo.add_edge(origem, destino, weight=peso)
    return grafo


def extrair_subgrafo_filtrado(
    grafo: nx.DiGraph,
    df: pd.DataFrame,
    instituicao: str,
    turno: str = "TODOS",
) -> nx.DiGraph:
    """Extrai subgrafo induzido pelos filtros de instituicao e turno."""
    df_filtrado = df.copy()
    if instituicao != "GERAL":
        df_filtrado = df_filtrado[df_filtrado["instituicao"] == instituicao]
    if turno != "TODOS":
        df_filtrado = df_filtrado[df_filtrado["turno"] == turno]
    ids_selecionados = df_filtrado["id_aluna"].tolist()
    return grafo.subgraph(ids_selecionados).copy()


def _converter_para_nao_direcionado_ponderado(grafo: nx.DiGraph) -> nx.Graph:
    """Converte grafo direcionado em grafo simples nao direcionado somando pesos."""
    grafo_nao_dir = nx.Graph()
    grafo_nao_dir.add_nodes_from(grafo.nodes(data=True))
    for u, v, data in grafo.edges(data=True):
        peso = float(data.get("weight", 1.0))
        if grafo_nao_dir.has_edge(u, v):
            grafo_nao_dir[u][v]["weight"] += peso
        else:
            grafo_nao_dir.add_edge(u, v, weight=peso)
    return grafo_nao_dir


def _adicionar_distancia_inversa_da_forca(grafo: nx.Graph) -> nx.Graph:
    """Copia o grafo e define distância como inverso da força positiva da aresta."""
    grafo_com_distancias = grafo.copy()
    for origem, destino, dados in grafo_com_distancias.edges(data=True):
        forca = float(dados.get("weight", 1.0))
        if not np.isfinite(forca) or forca <= 0:
            raise ValueError(
                f"A força da aresta ({origem}, {destino}) deve ser positiva e finita."
            )
        dados["distance"] = 1.0 / forca
    return grafo_com_distancias


def _selecionar_particao_girvan_newman(
    grafo_nao_dir: nx.Graph, max_comunidades: int = 6
) -> List[Set[str]]:
    """Seleciona particao por modularidade ponderada com cortes por distancia inversa."""
    grafo_com_distancias = _adicionar_distancia_inversa_da_forca(grafo_nao_dir)

    def aresta_mais_valiosa(grafo_atual: nx.Graph) -> Tuple[str, str]:
        centralidade = nx.edge_betweenness_centrality(grafo_atual, weight="distance")
        return max(centralidade, key=centralidade.get)

    comp_generator = nx.community.girvan_newman(
        grafo_com_distancias,
        most_valuable_edge=aresta_mais_valiosa,
    )
    melhor_particao: Optional[List[Set[str]]] = None
    melhor_mod = -1.0
    limite_comunidades = min(max_comunidades, grafo_nao_dir.number_of_nodes())
    for particao in comp_generator:
        comunidades_atuais = [set(c) for c in particao]
        if len(comunidades_atuais) > limite_comunidades:
            break
        mod_atual = nx.community.modularity(
            grafo_nao_dir, comunidades_atuais, weight="weight"
        )
        if melhor_particao is None or mod_atual > melhor_mod:
            melhor_mod = mod_atual
            melhor_particao = comunidades_atuais
        if len(comunidades_atuais) == limite_comunidades:
            break
    if melhor_particao is None:
        return [set(grafo_nao_dir.nodes())]
    return melhor_particao


def detectar_comunidades(
    grafo: nx.DiGraph, algoritmo: str = "louvain", seed: int = 42
) -> Tuple[List[Set[str]], float, str]:
    """Detecta comunidades pelo algoritmo escolhido e retorna (comunidades, modularidade, nome_alg)."""
    grafo_nao_dir = _converter_para_nao_direcionado_ponderado(grafo)
    if grafo_nao_dir.number_of_edges() == 0:
        comunidades_isoladas = [{n} for n in grafo_nao_dir.nodes()]
        return comunidades_isoladas, 0.0, algoritmo

    if algoritmo == "louvain":
        comunidades_raw = nx.community.louvain_communities(grafo_nao_dir, weight="weight", seed=seed)
        comunidades = [set(c) for c in comunidades_raw]
        nome_alg = "Louvain"
    elif algoritmo == "label_propagation":
        comunidades_raw = nx.community.asyn_lpa_communities(
            grafo_nao_dir,
            weight="weight",
            seed=seed,
        )
        comunidades = [set(c) for c in comunidades_raw]
        nome_alg = "Label Propagation"
    elif algoritmo == "greedy_modularity":
        comunidades_raw = nx.community.greedy_modularity_communities(grafo_nao_dir, weight="weight")
        comunidades = [set(c) for c in comunidades_raw]
        nome_alg = "Greedy Modularity"
    elif algoritmo == "girvan_newman":
        comunidades = _selecionar_particao_girvan_newman(grafo_nao_dir)
        nome_alg = "Girvan-Newman"
    else:
        comunidades_raw = nx.community.louvain_communities(grafo_nao_dir, weight="weight", seed=seed)
        comunidades = [set(c) for c in comunidades_raw]
        nome_alg = "Louvain"

    modularidade = nx.community.modularity(grafo_nao_dir, comunidades, weight="weight")
    return comunidades, modularidade, nome_alg


def calcular_proporcao_conexoes_mesmo_turno(grafo: nx.DiGraph) -> float:
    """Calcula o percentual de arestas direcionadas com turnos conhecidos iguais."""
    mesmo_turno = 0
    arestas_com_turno_conhecido = 0
    for u, v in grafo.edges():
        turno_u = grafo.nodes[u].get("turno", "")
        turno_v = grafo.nodes[v].get("turno", "")
        if (
            pd.notna(turno_u)
            and str(turno_u).strip()
            and pd.notna(turno_v)
            and str(turno_v).strip()
        ):
            arestas_com_turno_conhecido += 1
        else:
            continue
        if turno_u == turno_v:
            mesmo_turno += 1
    if arestas_com_turno_conhecido == 0:
        return 0.0
    return round((mesmo_turno / arestas_com_turno_conhecido) * 100.0, 1)


def calcular_metricas_rede(
    grafo: nx.DiGraph, comunidades: List[Set[str]], modularidade: float
) -> Dict[str, object]:
    """Calcula métricas da rede usando distância inversa para betweenness ponderado."""
    graus_entrada = dict(grafo.in_degree())
    principal_hub = max(graus_entrada, key=graus_entrada.get) if graus_entrada else None
    nome_hub = grafo.nodes[principal_hub].get("nome", "N/A") if principal_hub else "N/A"

    grafo_com_distancias = _adicionar_distancia_inversa_da_forca(grafo)
    betweenness = (
        nx.betweenness_centrality(grafo_com_distancias, weight="distance")
        if grafo.number_of_nodes() > 0
        else {}
    )
    ponte_id = max(betweenness, key=betweenness.get) if betweenness else None
    nome_ponte = grafo.nodes[ponte_id].get("nome", "N/A") if ponte_id else "N/A"

    grau_medio = (
        sum(dict(grafo.degree()).values()) / grafo.number_of_nodes()
        if grafo.number_of_nodes() > 0
        else 0.0
    )
    proporcao_conexoes_mesmo_turno = calcular_proporcao_conexoes_mesmo_turno(grafo)

    return {
        "num_nos": grafo.number_of_nodes(),
        "num_arestas": grafo.number_of_edges(),
        "densidade": round(nx.density(grafo), 4),
        "grau_medio": round(grau_medio, 2),
        "num_comunidades": len(comunidades),
        "modularidade": round(modularidade, 4),
        "proporcao_conexoes_mesmo_turno_pct": proporcao_conexoes_mesmo_turno,
        "principal_hub": f"{principal_hub} ({nome_hub})" if principal_hub else "N/A",
        "grau_entrada_max": graus_entrada.get(principal_hub, 0) if principal_hub else 0,
        "principal_ponte": f"{ponte_id} ({nome_ponte})" if ponte_id else "N/A",
        "betweenness_max": round(betweenness.get(ponte_id, 0.0), 3) if ponte_id else 0.0,
    }


def gerar_resumo_comunidades(
    grafo: nx.DiGraph, comunidades: List[Set[str]]
) -> List[Dict[str, object]]:
    """Gera tabela resumo por comunidade ordenada por quantidade de participantes."""
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
