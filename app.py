"""Ponto de entrada da aplicacao Dash SWITAB (Pareamento e Comunidades)."""

from functools import lru_cache
import hashlib
import io
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html
import networkx as nx
import pandas as pd

from components.cards import criar_card_metrica, gerar_resumo_textual_complementar
from config.settings import obter_configuracao
from core.comunidades_engine import (
    ALGORITMOS_DISPONIVEIS,
    calcular_metricas_rede,
    carregar_dados_comunidades,
    construir_grafo_completo,
    detectar_comunidades,
    extrair_subgrafo_filtrado,
    gerar_resumo_comunidades,
)
from core.pareamento_engine import (
    aplicar_flag_todas_alunas,
    carregar_dados_base_com_relatorio,
    construir_caracteristicas_gerais_rede,
    construir_grafo_e_metricas,
    construir_matriz_compatibilidade,
    executar_ag_mono_ou_elitista,
    executar_gulosa_afinidade_balanceada,
    executar_nsga2,
    executar_pareamento_peso_maximo_bipartido,
    gerar_rede_aleatoria,
    normalizar_chave_comparacao,
    preparar_conjuntos_bipartidos,
)
from pages.comunidades_page import criar_painel_comunidades
from pages.pareamento_page import criar_painel_pareamento
from viz.graficos_comunidades import criar_figura_grafo
from viz.graficos_pareamento import gerar_grafico_similaridade_pares

RAIZ_PROJETO = os.path.dirname(os.path.abspath(__file__))
CONFIG = obter_configuracao(RAIZ_PROJETO)

df_global, relatorio_global = carregar_dados_base_com_relatorio(
    caminho_arquivo=CONFIG.caminho_dados_pareamento,
)
instituicoes_individuais = sorted(df_global["instituicao"].dropna().astype(str).str.strip().unique().tolist())
opcoes_filtro_geral_inst = [{"label": "Todas as Instituições", "value": "TODAS"}] + [
    {"label": inst, "value": inst} for inst in instituicoes_individuais
]

df_comunidades, _ = carregar_dados_comunidades(
    caminho_csv=CONFIG.caminho_dados_comunidades,
)
inst_comunidades_disp = sorted(df_comunidades["instituicao"].dropna().unique().tolist())
turnos_comunidades_disp = sorted(df_comunidades["turno"].dropna().unique().tolist())

opcoes_dropdown_inst_com = [
    {"label": "Instituição Geral (Rede Completa)", "value": "GERAL"}
] + [{"label": inst, "value": inst} for inst in inst_comunidades_disp]

opcoes_dropdown_turno_com = [
    {"label": "Todos os Turnos", "value": "TODOS"}
] + [{"label": t, "value": t} for t in turnos_comunidades_disp]

opcoes_filtro_tabela_com = [
    {"label": "Todas as Instituições + Instituição Geral", "value": "TODAS_LINHAS"},
    {"label": "Apenas Instituição Geral (Rede Completa)", "value": "GERAL"},
] + [{"label": inst, "value": inst} for inst in inst_comunidades_disp]

_CACHE_SIMULACAO: Dict[str, Tuple[nx.Graph, Dict[str, Any], Dict[str, Dict[str, Any]]]] = {}


def _assinatura_simulacao(
    inst_lista: List[str],
    cursos_lista: List[str],
    modelo: str,
    papeis_sel: Optional[List[str]],
    regra_inst: bool,
    regra_curso: bool,
    regra_periodo: bool,
    forcar_todas_alunas: bool,
    max_par: int,
    pesos_afinidade: Tuple[float, float, float],
    penalidades: Tuple[float, float, float],
) -> str:
    """Gera chave SHA-256 unica para cachear resultados de simulacao de pareamento."""
    chave = (
        f"{tuple(sorted(inst_lista))}|{tuple(sorted(cursos_lista))}|{modelo}|"
        f"{tuple(sorted(papeis_sel or []))}|{regra_inst}|{regra_curso}|{regra_periodo}|"
        f"{forcar_todas_alunas}|{max_par}|{pesos_afinidade}|{penalidades}"
    ).encode("utf-8")
    return hashlib.sha256(chave).hexdigest()


def rodar_simulacao_cacheada(
    df: pd.DataFrame,
    inst_lista: List[str],
    cursos_lista: List[str],
    modelo: str,
    papeis_sel: Optional[List[str]],
    regra_inst: bool,
    regra_curso: bool,
    regra_periodo: bool,
    forcar_todas_alunas: bool,
    max_par: int,
    pesos_afinidade: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    penalidades: Tuple[float, float, float] = (2.5, 1.5, 0.8),
) -> Tuple[nx.Graph, Dict[str, Any], Dict[str, Dict[str, Any]], int]:
    """Executa o algoritmo de pareamento selecionado com cache em memoria."""
    max_par = max(1, int(max_par))
    mentoras, mentoradas = preparar_conjuntos_bipartidos(
        df,
        papeis_selecionados=papeis_sel,
        instituicoes_selecionadas=inst_lista,
        cursos_selecionados=cursos_lista,
    )
    total_v = len(mentoradas)
    if len(mentoras) == 0 or total_v == 0:
        return nx.Graph(), {}, {}, total_v

    chave = _assinatura_simulacao(
        inst_lista,
        cursos_lista,
        modelo,
        papeis_sel,
        regra_inst,
        regra_curso,
        regra_periodo,
        forcar_todas_alunas,
        max_par,
        pesos_afinidade,
        penalidades,
    )
    if chave in _CACHE_SIMULACAO:
        G_c, m_ger_c, m_inst_c = _CACHE_SIMULACAO[chave]
        return G_c, m_ger_c, m_inst_c, total_v

    matrizes = construir_matriz_compatibilidade(mentoras, mentoradas, pesos_afinidade=pesos_afinidade)

    if modelo == "aleatoria":
        aloc_bruta = gerar_rede_aleatoria(
            mentoras,
            mentoradas,
            matrizes,
            regra_inst,
            regra_curso,
            regra_periodo,
            max_par=max_par,
            forcar_todas_alunas=forcar_todas_alunas,
            seed=CONFIG.seed,
        )
    elif modelo == "ag_mono":
        aloc_bruta = executar_ag_mono_ou_elitista(
            mentoras,
            mentoradas,
            matrizes,
            regra_inst,
            regra_curso,
            regra_periodo,
            elitismo=False,
            max_par=max_par,
            forcar_todas_alunas=forcar_todas_alunas,
            penalidades=penalidades,
            seed=CONFIG.seed,
        )
    elif modelo == "ag_elitista":
        aloc_bruta = executar_ag_mono_ou_elitista(
            mentoras,
            mentoradas,
            matrizes,
            regra_inst,
            regra_curso,
            regra_periodo,
            elitismo=True,
            max_par=max_par,
            forcar_todas_alunas=forcar_todas_alunas,
            penalidades=penalidades,
            seed=CONFIG.seed,
        )
    elif modelo == "bipartite_max_matching":
        aloc_bruta = executar_pareamento_peso_maximo_bipartido(
            mentoras,
            mentoradas,
            matrizes,
            regra_inst=regra_inst,
            regra_curso=regra_curso,
            regra_periodo=regra_periodo,
            max_par=max_par,
            forcar_todas_alunas=forcar_todas_alunas,
        )
    elif modelo == "gulosa_afinidade":
        aloc_bruta = executar_gulosa_afinidade_balanceada(
            mentoras,
            mentoradas,
            matrizes,
            regra_inst=regra_inst,
            regra_curso=regra_curso,
            regra_periodo=regra_periodo,
            max_par=max_par,
            forcar_todas_alunas=forcar_todas_alunas,
        )
    else:
        aloc_bruta = executar_nsga2(
            mentoras,
            mentoradas,
            matrizes,
            regra_inst,
            regra_curso,
            regra_periodo,
            max_par=max_par,
            forcar_todas_alunas=forcar_todas_alunas,
            penalidades=penalidades,
            seed=CONFIG.seed,
        )

    aloc = aplicar_flag_todas_alunas(
        aloc_bruta,
        mentoras,
        mentoradas,
        matrizes,
        regra_inst,
        regra_curso,
        regra_periodo,
        max_par,
        forcar_todas_alunas,
    )

    G_con, met_geral, met_por_inst = construir_grafo_e_metricas(
        mentoras,
        mentoradas,
        aloc,
        matrizes,
        max_par=max_par,
    )

    if len(_CACHE_SIMULACAO) >= 256:
        _CACHE_SIMULACAO.clear()
    _CACHE_SIMULACAO[chave] = (G_con, met_geral, met_por_inst)
    return G_con, met_geral, met_por_inst, total_v


@lru_cache(maxsize=64)
def _calcular_tabela_comparativa_cached(
    criterio_peso: str, filtro_instituicao: str
) -> Tuple[Dict[str, object], ...]:
    """Calcula tabela comparativa de algoritmos de comunidades por instituicao."""
    grafo_global = construir_grafo_completo(df_comunidades, criterio_peso=criterio_peso)
    escopos = ["GERAL"] + sorted(df_comunidades["instituicao"].dropna().unique().tolist())
    if filtro_instituicao != "TODAS_LINHAS":
        escopos = [filtro_instituicao]

    linhas: List[Dict[str, object]] = []
    for escopo in escopos:
        rotulo_inst = "Instituição Geral (Rede Completa)" if escopo == "GERAL" else escopo
        subgrafo = extrair_subgrafo_filtrado(grafo_global, df_comunidades, escopo, turno="TODOS")
        for alg_info in ALGORITMOS_DISPONIVEIS:
            alg_cod = alg_info["value"]
            comunidades, modularidade, nome_alg = detectar_comunidades(
                subgrafo, algoritmo=alg_cod, seed=CONFIG.seed
            )
            metricas = calcular_metricas_rede(subgrafo, comunidades, modularidade)
            linhas.append({
                "instituicao": rotulo_inst,
                "algoritmo": nome_alg,
                "num_nos": metricas["num_nos"],
                "num_arestas": metricas["num_arestas"],
                "densidade": metricas["densidade"],
                "grau_medio": metricas["grau_medio"],
                "proporcao_conexoes_mesmo_turno_pct": metricas[
                    "proporcao_conexoes_mesmo_turno_pct"
                ],
                "num_comunidades": metricas["num_comunidades"],
                "modularidade": metricas["modularidade"],
                "principal_hub": metricas["principal_hub"],
            })
    return tuple(linhas)


def gerar_tabela_comparativa(
    criterio_peso: str = "composto", filtro_instituicao: str = "TODAS_LINHAS"
) -> List[Dict[str, object]]:
    """Retorna lista de registros da tabela comparativa de comunidades."""
    return list(_calcular_tabela_comparativa_cached(criterio_peso, filtro_instituicao))


painel_pareamento = criar_painel_pareamento(
    opcoes_filtro_geral_inst=opcoes_filtro_geral_inst,
    config=CONFIG,
)

painel_comunidades = criar_painel_comunidades(
    opcoes_dropdown_inst_com=opcoes_dropdown_inst_com,
    opcoes_dropdown_turno_com=opcoes_dropdown_turno_com,
    opcoes_filtro_tabela_com=opcoes_filtro_tabela_com,
)

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="SWITAB - Dashboard de Pareamento e Comunidades",
)

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2(
                            "SWITAB - Dashboard",
                            style={"fontWeight": "700", "color": "#4a148c", "marginTop": "18px", "marginBottom": "4px"},
                        ),
                        html.P(
                            "Selecione o painel desejado entre Pareamento e Comunidades.",
                            style={"color": "#495057", "fontSize": "0.95rem", "marginBottom": "12px"},
                        ),
                        dbc.Tabs(
                            id="tabs-paineis-switab",
                            active_tab="painel-pareamento",
                            children=[
                                dbc.Tab(label="Pareamento", tab_id="painel-pareamento"),
                                dbc.Tab(label="Comunidades", tab_id="painel-comunidades"),
                            ],
                            style={"marginBottom": "8px"},
                        ),
                    ]
                )
            ]
        ),
        html.Div(
            [
                html.Div(painel_pareamento, id="wrapper-painel-pareamento", style={"display": "block"}),
                html.Div(painel_comunidades, id="wrapper-painel-comunidades", style={"display": "none"}),
            ]
        ),
    ],
    fluid=True,
    style={"backgroundColor": "#f4f6f9", "minHeight": "100vh", "paddingBottom": "32px"},
)


@app.callback(
    [
        Output("wrapper-painel-pareamento", "style"),
        Output("wrapper-painel-comunidades", "style"),
    ],
    Input("tabs-paineis-switab", "active_tab"),
)
def alternar_paineis(aba_ativa: Optional[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Alterna a visibilidade entre os paineis Pareamento e Comunidades."""
    if aba_ativa == "painel-comunidades":
        return {"display": "none"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}


@app.callback(
    Output("collapse-personalizacao", "is_open"),
    Input("btn-toggle-personalizacao", "n_clicks"),
    State("collapse-personalizacao", "is_open"),
    prevent_initial_call=True,
)
def toggle_personalizacao(n_clicks: Optional[int], is_open: bool) -> bool:
    """Abre ou fecha o bloco de parametros personalizados de afinidade e penalidades."""
    return not is_open


@app.callback(
    Output("collapse-caracteristicas", "is_open"),
    Input("btn-toggle-caracteristicas", "n_clicks"),
    State("collapse-caracteristicas", "is_open"),
    prevent_initial_call=True,
)
def toggle_caracteristicas(n_clicks: Optional[int], is_open: bool) -> bool:
    """Abre ou fecha a tabela de caracteristicas gerais da rede."""
    return not is_open


@app.callback(
    Output("collapse-explicacao", "is_open"),
    Input("btn-toggle-explicacao", "n_clicks"),
    State("collapse-explicacao", "is_open"),
    prevent_initial_call=True,
)
def toggle_explicacao(n_clicks: Optional[int], is_open: bool) -> bool:
    """Abre ou fecha a explicacao didatica dos algoritmos de pareamento."""
    return not is_open


@app.callback(
    Output("collapse-explicacao-metricas", "is_open"),
    Input("btn-toggle-explicacao-metricas", "n_clicks"),
    State("collapse-explicacao-metricas", "is_open"),
    prevent_initial_call=True,
)
def toggle_explicacao_metricas(n_clicks: Optional[int], is_open: bool) -> bool:
    """Abre ou fecha a explicacao de cada metrica do painel Pareamento."""
    return not is_open


@app.callback(
    Output("collapse-sugestoes", "is_open"),
    Input("btn-toggle-sugestoes", "n_clicks"),
    State("collapse-sugestoes", "is_open"),
    prevent_initial_call=True,
)
def toggle_sugestoes(n_clicks: Optional[int], is_open: bool) -> bool:
    """Abre ou fecha o bloco de sugestoes de configuracao de filtros."""
    return not is_open


@app.callback(
    Output("collapse-pares", "is_open"),
    Input("btn-toggle-pares", "n_clicks"),
    State("collapse-pares", "is_open"),
    prevent_initial_call=True,
)
def toggle_pares(n_clicks: Optional[int], is_open: bool) -> bool:
    """Abre ou fecha a tabela de pares formados e exportacao CSV."""
    return not is_open


@app.callback(
    Output("download-csv", "data"),
    Input("btn-exportar-csv", "n_clicks"),
    State("tabela-pares", "data"),
    State("dropdown-modelo", "value"),
    prevent_initial_call=True,
)
def exportar_csv(n_clicks: Optional[int], dados_tabela: Optional[List[Dict[str, Any]]], modelo: Optional[str]) -> Any:
    """Exporta os pares formados para arquivo CSV."""
    if not dados_tabela:
        return None
    df_out = pd.DataFrame(dados_tabela)
    nome_modelo = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(modelo or "pareamento"))
    return dcc.send_data_frame(df_out.to_csv, f"pares_switab_{nome_modelo}.csv", index=False, encoding="utf-8-sig")


@app.callback(
    [
        Output("check-cursos", "options"),
        Output("check-cursos", "value"),
    ],
    Input("check-instituicoes", "value"),
)
def atualizar_opcoes_cursos(inst_flags_sel: Any) -> Tuple[List[Dict[str, str]], List[str]]:
    """Atualiza opcoes de cursos disponiveis de acordo com as instituicoes selecionadas."""
    inst_lista = inst_flags_sel if isinstance(inst_flags_sel, list) else ([inst_flags_sel] if inst_flags_sel else ["TODAS"])
    inst_clean = [str(x).strip() for x in inst_lista if str(x).strip()]
    individuais = [x for x in inst_clean if x != "TODAS"]

    df_sub = df_global.copy()
    if individuais:
        alvo_inst = {normalizar_chave_comparacao(x) for x in individuais}
        df_sub = df_sub[df_sub["instituicao"].apply(lambda v: normalizar_chave_comparacao(v) in alvo_inst)]

    cursos_inst = sorted(df_sub["curso"].dropna().astype(str).str.strip().unique().tolist())
    opcoes = [{"label": " Selecionar Todos", "value": "TODOS_CURSOS"}] + [
        {"label": f" {c}", "value": c} for c in cursos_inst
    ]
    return opcoes, ["TODOS_CURSOS"]


def montar_linhas_metricas_isoladas(
    df: pd.DataFrame,
    modelo: str,
    papeis_sel: Optional[List[str]],
    regra_inst: bool,
    regra_curso: bool,
    regra_periodo: bool,
    forcar_todas_alunas: bool,
    max_par: int,
    pesos_afinidade: Tuple[float, float, float],
    penalidades: Tuple[float, float, float],
    flags_metricas_inst: Any = None,
) -> List[Dict[str, Any]]:
    """Monta linhas de metricas gerais e institucionais filtradas por flags."""
    _, met_geral_todas, met_por_inst_todas, _ = rodar_simulacao_cacheada(
        df=df,
        inst_lista=["TODAS"],
        cursos_lista=["TODOS_CURSOS"],
        modelo=modelo,
        papeis_sel=papeis_sel,
        regra_inst=regra_inst,
        regra_curso=regra_curso,
        regra_periodo=regra_periodo,
        forcar_todas_alunas=forcar_todas_alunas,
        max_par=max_par,
        pesos_afinidade=pesos_afinidade,
        penalidades=penalidades,
    )
    todas_linhas: List[Dict[str, Any]] = []
    if met_geral_todas:
        todas_linhas.append(met_geral_todas)
    todas_linhas.extend(met_por_inst_todas.values())

    if not todas_linhas:
        return []

    chaves_alvo = set()
    flags_list = flags_metricas_inst if isinstance(flags_metricas_inst, list) else ([flags_metricas_inst] if flags_metricas_inst else ["TODAS"])
    flags_ind = [str(x).strip() for x in flags_list if str(x).strip() and str(x).strip() != "TODAS"]
    for f_inst in flags_ind:
        chaves_alvo.add(normalizar_chave_comparacao(f_inst))

    if not chaves_alvo:
        return todas_linhas

    filtradas = [
        linha for linha in todas_linhas
        if normalizar_chave_comparacao(str(linha.get("Escopo", ""))) in chaves_alvo
    ]
    return filtradas if filtradas else todas_linhas


@app.callback(
    Output("container-tabela-metricas", "style"),
    Input("flag-exibir-metricas-acima-csv", "value"),
)
def toggle_exibir_metricas_acima_csv(flag_val: Any) -> Dict[str, str]:
    """Alterna exibicao do container da tabela de metricas."""
    ativos = flag_val if isinstance(flag_val, list) else ([flag_val] if flag_val else [])
    if "exibir_acima_csv" in ativos:
        return {"display": "block"}
    return {"display": "none"}


def _parse_float_seguro(val: Any, padrao: float) -> float:
    """Converte entrada numerica do usuario em float nao negativo com fallback."""
    if val is None or val == "":
        return padrao
    try:
        num = float(val)
        return num if num >= 0.0 else padrao
    except (ValueError, TypeError):
        return padrao


@app.callback(
    [
        Output("grafico-pareamento", "figure"),
        Output("resumo-textual-complementar", "children"),
        Output("tabela-metricas", "data"),
        Output("tabela-pares", "data"),
        Output("tabela-caracteristicas-rede", "data"),
    ],
    [
        Input("check-instituicoes", "value"),
        Input("check-cursos", "value"),
        Input("dropdown-tipo-grafico", "value"),
        Input("dropdown-modelo", "value"),
        Input("check-papeis", "value"),
        Input("check-regras", "value"),
        Input("flag-combinacao-todas-alunas", "value"),
        Input("slider-limite-par", "value"),
        Input("check-metricas-instituicoes", "value"),
        Input("input-peso-areas", "value"),
        Input("input-peso-instituicao", "value"),
        Input("input-peso-curso", "value"),
        Input("input-pen-periodo", "value"),
        Input("input-pen-instituicao", "value"),
        Input("input-pen-curso-carga", "value"),
    ],
)
def atualizar_dashboard_pareamento(
    inst_flags_sel: Any,
    cursos_flags_sel: Any,
    tipo_grafico: Optional[str],
    modelo: Optional[str],
    papeis_sel: Optional[List[str]],
    regras_sel: Optional[List[str]],
    flag_todas_sel: Any,
    max_par: Optional[int],
    flags_metricas_inst: Any,
    peso_areas_val: Any,
    peso_inst_val: Any,
    peso_curso_val: Any,
    pen_periodo_val: Any,
    pen_inst_val: Any,
    pen_curso_carga_val: Any,
) -> Tuple[Any, html.Div, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Atualiza grafico, resumo textual, metricas, pares e caracteristicas do painel Pareamento."""
    inst_lista = inst_flags_sel if isinstance(inst_flags_sel, list) else ([inst_flags_sel] if inst_flags_sel else ["TODAS"])
    cursos_lista = cursos_flags_sel if isinstance(cursos_flags_sel, list) else ([cursos_flags_sel] if cursos_flags_sel else ["TODOS_CURSOS"])
    tipo_graf = tipo_grafico or "similaridade_espacial"
    alg = modelo or "nsga2"
    limite_par = max(1, int(max_par or CONFIG.limite_conexoes_padrao))

    pesos_afinidade = (
        _parse_float_seguro(peso_areas_val, CONFIG.peso_areas),
        _parse_float_seguro(peso_inst_val, CONFIG.peso_instituicao),
        _parse_float_seguro(peso_curso_val, CONFIG.peso_curso),
    )
    penalidades = (
        _parse_float_seguro(pen_periodo_val, CONFIG.penalidade_periodo),
        _parse_float_seguro(pen_inst_val, CONFIG.penalidade_instituicao),
        _parse_float_seguro(pen_curso_carga_val, CONFIG.penalidade_curso_carga),
    )

    regra_periodo = "periodo" in (regras_sel or [])
    regra_inst = "inst" in (regras_sel or [])
    regra_curso = "curso" in (regras_sel or [])
    forcar_todas_alunas = bool(flag_todas_sel) and (
        "todas_alunas" in (flag_todas_sel if isinstance(flag_todas_sel, list) else [flag_todas_sel])
    )

    caract_linhas = construir_caracteristicas_gerais_rede(df_global, inst_lista, cursos_lista)

    G_con, _, _, total_mentoradas = rodar_simulacao_cacheada(
        df=df_global,
        inst_lista=inst_lista,
        cursos_lista=cursos_lista,
        modelo=alg,
        papeis_sel=papeis_sel,
        regra_inst=regra_inst,
        regra_curso=regra_curso,
        regra_periodo=regra_periodo,
        forcar_todas_alunas=forcar_todas_alunas,
        max_par=limite_par,
        pesos_afinidade=pesos_afinidade,
        penalidades=penalidades,
    )

    fig_par = gerar_grafico_similaridade_pares(G_con, tipo_grafico=tipo_graf, max_par=limite_par)
    resumo_txt = gerar_resumo_textual_complementar(G_con, total_mentoradas, relatorio_ingestao=relatorio_global)

    if G_con.number_of_edges() == 0:
        return fig_par, resumo_txt, [], [], caract_linhas

    linhas_metricas = montar_linhas_metricas_isoladas(
        df=df_global,
        modelo=alg,
        papeis_sel=papeis_sel,
        regra_inst=regra_inst,
        regra_curso=regra_curso,
        regra_periodo=regra_periodo,
        forcar_todas_alunas=forcar_todas_alunas,
        max_par=limite_par,
        pesos_afinidade=pesos_afinidade,
        penalidades=penalidades,
        flags_metricas_inst=flags_metricas_inst,
    )

    registros_pares: List[Dict[str, Any]] = []
    for u, v, d in G_con.edges(data=True):
        m_node = u if G_con.nodes[u]["grupo_bipartido"] == "U" else v
        a_node = v if G_con.nodes[u]["grupo_bipartido"] == "U" else u
        registros_pares.append(
            {
                "Mentora (U)": m_node,
                "Inst. Mentora": G_con.nodes[m_node]["instituicao"],
                "Curso Mentora": G_con.nodes[m_node]["curso"],
                "Ingresso U": G_con.nodes[m_node]["ano_ingresso"],
                "Participante (V)": a_node,
                "Papel V": G_con.nodes[a_node]["papel"],
                "Inst. V": G_con.nodes[a_node]["instituicao"],
                "Curso V": G_con.nodes[a_node]["curso"],
                "Ingresso V": G_con.nodes[a_node]["ano_ingresso"],
                "Afinidade": round(float(d["afinidade"]), 4),
                "Status": d["status"],
            }
        )

    return fig_par, resumo_txt, linhas_metricas, registros_pares, caract_linhas


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
def atualizar_grafo_e_cards_comunidades(
    instituicao_selecionada: str,
    turno_selecionado: str,
    algoritmo_selecionado: str,
    criterio_peso: str,
    exibir_setas_val: str,
):
    """Atualiza grafo, cards de metricas e resumo de comunidades no painel Comunidades."""
    grafo_ponderado = construir_grafo_completo(df_comunidades, criterio_peso=criterio_peso)
    subgrafo = extrair_subgrafo_filtrado(
        grafo_ponderado, df_comunidades, instituicao_selecionada, turno=turno_selecionado
    )
    comunidades, modularidade, nome_alg = detectar_comunidades(
        subgrafo, algoritmo=algoritmo_selecionado, seed=CONFIG.seed
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
        criar_card_metrica(
            "Conexões no Mesmo Turno",
            f"{metricas['proporcao_conexoes_mesmo_turno_pct']}%",
            "Percentual das arestas com turno conhecido",
        ),
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
def atualizar_tabela_comparativa_comunidades(filtro_tabela: str, criterio_peso: str):
    """Atualiza a tabela comparativa de algoritmos por instituicao."""
    return gerar_tabela_comparativa(
        criterio_peso=criterio_peso, filtro_instituicao=filtro_tabela
    )


@app.callback(
    Output("download-tabela-csv", "data"),
    [Input("btn-exportar-csv-comunidades", "n_clicks")],
    [
        State("filtro-tabela-instituicao", "value"),
        State("filtro-peso", "value"),
    ],
    prevent_initial_call=True,
)
def exportar_tabela_csv_comunidades(n_clicks: int, filtro_tabela: str, criterio_peso: str):
    """Exporta a tabela comparativa de comunidades para arquivo CSV."""
    dados = gerar_tabela_comparativa(
        criterio_peso=criterio_peso, filtro_instituicao=filtro_tabela
    )
    df_export = pd.DataFrame(dados)
    buffer = io.StringIO()
    df_export.to_csv(buffer, index=False, encoding="utf-8-sig")
    return dict(content=buffer.getvalue(), filename="comparativo_algoritmos_switab.csv")


if __name__ == "__main__":
    app.run(debug=CONFIG.debug, host=CONFIG.host, port=CONFIG.porta)
