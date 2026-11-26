"""Layout declarativo do painel Comunidades."""

from typing import Dict, List

import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

from core.comunidades_engine import ALGORITMOS_DISPONIVEIS, CRITERIOS_PESO


def criar_painel_comunidades(
    opcoes_dropdown_inst_com: List[Dict[str, str]],
    opcoes_dropdown_turno_com: List[Dict[str, str]],
    opcoes_filtro_tabela_com: List[Dict[str, str]],
) -> dbc.Container:
    """Constroi o container do painel Comunidades com filtros, grafo, cards e tabelas."""
    colunas_tabela_comparativa_com = [
        {"name": "Instituição", "id": "instituicao"},
        {"name": "Algoritmo", "id": "algoritmo"},
        {"name": "Nós", "id": "num_nos"},
        {"name": "Arestas", "id": "num_arestas"},
        {"name": "Densidade", "id": "densidade"},
        {"name": "Grau Médio", "id": "grau_medio"},
        {
            "name": "Arestas no Mesmo Turno (%)",
            "id": "proporcao_conexoes_mesmo_turno_pct",
        },
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

    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4(
                                "Comunidades",
                                style={"fontWeight": "700", "color": "#4a148c", "marginTop": "14px", "marginBottom": "4px"},
                            ),
                            html.P(
                                "Explore por instituição ou rede geral, compare algoritmos de comunidades, inspecione clusters e exporte métricas.",
                                style={"color": "#495057", "fontSize": "0.95rem", "marginBottom": "14px"},
                            ),
                        ]
                    )
                ]
            ),
            html.Div(
                style={
                    "backgroundColor": "#FFFFFF",
                    "padding": "16px 20px",
                    "borderRadius": "10px",
                    "border": "1px solid #E5E7EB",
                    "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
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
                                options=opcoes_dropdown_inst_com,
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
                                options=opcoes_dropdown_turno_com,
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
                style={"backgroundColor": "#FFFFFF", "borderRadius": "10px", "border": "1px solid #E5E7EB", "boxShadow": "0 2px 8px rgba(0,0,0,0.06)", "padding": "10px", "marginBottom": "20px"},
                children=[dcc.Graph(id="grafico-rede")],
            ),
            html.Div(
                style={"marginBottom": "24px"},
                children=[
                    html.H5(id="titulo-metricas", style={"fontWeight": "700", "color": "#4a148c", "marginBottom": "12px"}),
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
                    "borderRadius": "10px",
                    "border": "1px solid #E5E7EB",
                    "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
                    "marginBottom": "24px",
                },
                children=[
                    html.H5("Resumo das Comunidades Detectadas na Seleção Atual", style={"fontWeight": "700", "color": "#4a148c", "margin": "0 0 12px 0"}),
                    dash_table.DataTable(
                        id="tabela-resumo-comunidades",
                        columns=colunas_tabela_comunidades,
                        data=[],
                        sort_action="native",
                        page_size=10,
                        style_table={"overflowX": "auto"},
                        style_header={
                            "backgroundColor": "#4a148c",
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
                    "borderRadius": "10px",
                    "border": "1px solid #E5E7EB",
                    "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
                    "marginBottom": "24px",
                },
                children=[
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "flexWrap": "wrap", "gap": "16px", "marginBottom": "16px"},
                        children=[
                            html.Div(
                                children=[
                                    html.H5("Tabela Comparativa: Todos os Algoritmos por Instituição", style={"fontWeight": "700", "color": "#4a148c", "margin": "0 0 4px 0"}),
                                    html.P("Compare comunidades, modularidade (Q), proporção de arestas no mesmo turno e hubs por algoritmo e instituição.", style={"color": "#4B5563", "margin": "0", "fontSize": "13px"}),
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
                                                options=opcoes_filtro_tabela_com,
                                                value="TODAS_LINHAS",
                                                clearable=False,
                                            ),
                                        ],
                                    ),
                                    html.Button(
                                        "Exportar CSV",
                                        id="btn-exportar-csv-comunidades",
                                        n_clicks=0,
                                        style={
                                            "backgroundColor": "#198754",
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
                        columns=colunas_tabela_comparativa_com,
                        data=[],
                        sort_action="native",
                        filter_action="native",
                        page_size=24,
                        style_table={"overflowX": "auto"},
                        style_header={
                            "backgroundColor": "#4a148c",
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
        fluid=True,
        style={"paddingLeft": "0", "paddingRight": "0"},
    )
