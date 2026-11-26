"""Layout declarativo do painel Pareamento."""

from typing import Dict, List

import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

from config.settings import ConfiguracaoApp


def criar_painel_pareamento(
    opcoes_filtro_geral_inst: List[Dict[str, str]],
    config: ConfiguracaoApp,
) -> dbc.Container:
    """Constroi o container do painel Pareamento com filtros, Personalizacao, graficos e tabelas."""
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4(
                                "Pareamento",
                                style={"fontWeight": "700", "color": "#4a148c", "marginTop": "14px", "marginBottom": "4px"},
                            ),
                            html.P(
                                "Painel interativo para simular, otimizar e avaliar a formação de pares de mentoria na rede SWITAB, "
                                "comparando algoritmos, regras institucionais/acadêmicas, afinidade temática e carga de mentoras.",
                                style={"color": "#495057", "fontSize": "0.95rem", "marginBottom": "14px"},
                            ),
                        ]
                    )
                ]
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Tipo de Gráfico", style={"fontWeight": "600", "fontSize": "0.85rem"}),
                                        dcc.Dropdown(
                                            id="dropdown-tipo-grafico",
                                            options=[
                                                {"label": "Pares por Similaridade", "value": "similaridade_espacial"},
                                                {"label": "Grafo Bipartido", "value": "grafo_bipartido"},
                                            ],
                                            value="similaridade_espacial",
                                            clearable=False,
                                        ),
                                    ],
                                    md=3,
                                    className="mb-2",
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Instituição", style={"fontWeight": "600", "fontSize": "0.85rem"}),
                                        dbc.DropdownMenu(
                                            label="Selecionar Instituições",
                                            color="light",
                                            toggle_style={
                                                "width": "100%",
                                                "textAlign": "left",
                                                "border": "1px solid #ced4da",
                                                "color": "#212529",
                                            },
                                            children=[
                                                dbc.DropdownMenuItem(
                                                    dbc.Checklist(
                                                        id="check-instituicoes",
                                                        options=opcoes_filtro_geral_inst,
                                                        value=["TODAS"],
                                                    ),
                                                    toggle=False,
                                                )
                                            ],
                                        ),
                                    ],
                                    md=3,
                                    className="mb-2",
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Algoritmo", style={"fontWeight": "600", "fontSize": "0.85rem"}),
                                        dcc.Dropdown(
                                            id="dropdown-modelo",
                                            options=[
                                                {"label": "Rede Aleatória", "value": "aleatoria"},
                                                {"label": "Gulosa por Afinidade e Carga", "value": "gulosa_afinidade"},
                                                {"label": "Matching: Cardinalidade Máxima e Maior Peso", "value": "bipartite_max_matching"},
                                                {"label": "AG Clássico (Mono-objetivo)", "value": "ag_mono"},
                                                {"label": "AG Elitista", "value": "ag_elitista"},
                                                {"label": "NSGA-II (Multiobjetivo)", "value": "nsga2"},
                                            ],
                                            value="aleatoria",
                                            clearable=False,
                                        ),
                                    ],
                                    md=3,
                                    className="mb-2",
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Papéis", style={"fontWeight": "600", "fontSize": "0.85rem"}),
                                        dbc.DropdownMenu(
                                            label="Selecionar Papéis",
                                            color="light",
                                            toggle_style={
                                                "width": "100%",
                                                "textAlign": "left",
                                                "border": "1px solid #ced4da",
                                                "color": "#212529",
                                            },
                                            children=[
                                                dbc.DropdownMenuItem(
                                                    dbc.Checklist(
                                                        id="check-papeis",
                                                        options=[
                                                            {"label": " Mentora", "value": "Mentora"},
                                                            {"label": " Mentorada", "value": "Mentorada"},
                                                            {"label": " Embaixadora", "value": "Embaixadora"},
                                                        ],
                                                        value=["Mentora", "Mentorada"],
                                                    ),
                                                    toggle=False,
                                                )
                                            ],
                                        ),
                                    ],
                                    md=3,
                                    className="mb-2",
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Curso na Instituição", style={"fontWeight": "600", "fontSize": "0.85rem"}),
                                        dbc.DropdownMenu(
                                            label="Selecionar Cursos",
                                            color="light",
                                            toggle_style={
                                                "width": "100%",
                                                "textAlign": "left",
                                                "border": "1px solid #ced4da",
                                                "color": "#212529",
                                            },
                                            children=[
                                                dbc.DropdownMenuItem(
                                                    dbc.Checklist(
                                                        id="check-cursos",
                                                        options=[{"label": " Selecionar Todos", "value": "TODOS_CURSOS"}],
                                                        value=["TODOS_CURSOS"],
                                                    ),
                                                    toggle=False,
                                                )
                                            ],
                                        ),
                                    ],
                                    md=3,
                                    className="mb-2",
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Regras", style={"fontWeight": "600", "fontSize": "0.85rem"}),
                                        dbc.DropdownMenu(
                                            label="Selecionar Regras",
                                            color="light",
                                            toggle_style={
                                                "width": "100%",
                                                "textAlign": "left",
                                                "border": "1px solid #ced4da",
                                                "color": "#212529",
                                            },
                                            children=[
                                                dbc.DropdownMenuItem(
                                                    dbc.Checklist(
                                                        id="check-regras",
                                                        options=[
                                                            {"label": " Período (>1 sem.)", "value": "periodo"},
                                                            {"label": " Instituição", "value": "inst"},
                                                            {"label": " Curso", "value": "curso"},
                                                        ],
                                                        value=["periodo", "inst", "curso"],
                                                    ),
                                                    toggle=False,
                                                )
                                            ],
                                        ),
                                    ],
                                    md=3,
                                    className="mb-2",
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Limite de Conexões por Par", style={"fontWeight": "600", "fontSize": "0.85rem"}),
                                        html.Div(
                                            dcc.Slider(
                                                id="slider-limite-par",
                                                min=1,
                                                max=5,
                                                step=1,
                                                value=config.limite_conexoes_padrao,
                                                marks={i: str(i) for i in range(1, 6)},
                                                tooltip={"placement": "bottom", "always_visible": False},
                                            ),
                                            style={"paddingTop": "4px"},
                                        ),
                                    ],
                                    md=3,
                                    className="mb-2",
                                ),
                                dbc.Col(
                                    [
                                        html.Div(
                                            dbc.Checklist(
                                                id="flag-combinacao-todas-alunas",
                                                options=[{"label": " Combinação para todas as alunas", "value": "todas_alunas"}],
                                                value=["todas_alunas"],
                                                style={"fontWeight": "600", "fontSize": "0.9rem", "paddingTop": "22px"},
                                            )
                                        )
                                    ],
                                    md=3,
                                    className="mb-2",
                                ),
                            ]
                        ),
                        html.Hr(style={"margin": "10px 0 12px 0"}),
                        dbc.Button(
                            "Personalização",
                            id="btn-toggle-personalizacao",
                            color="secondary",
                            outline=True,
                            size="sm",
                            style={"fontWeight": "600"},
                        ),
                        dbc.Collapse(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label("Peso Áreas (Jaccard)", style={"fontWeight": "600", "fontSize": "0.83rem"}),
                                                        dcc.Input(
                                                            id="input-peso-areas",
                                                            type="number",
                                                            value=config.peso_areas,
                                                            step=0.05,
                                                            min=0.0,
                                                            max=10.0,
                                                            className="form-control form-control-sm",
                                                        ),
                                                    ],
                                                    md=2,
                                                    className="mb-2",
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Peso Instituição", style={"fontWeight": "600", "fontSize": "0.83rem"}),
                                                        dcc.Input(
                                                            id="input-peso-instituicao",
                                                            type="number",
                                                            value=config.peso_instituicao,
                                                            step=0.05,
                                                            min=0.0,
                                                            max=10.0,
                                                            className="form-control form-control-sm",
                                                        ),
                                                    ],
                                                    md=2,
                                                    className="mb-2",
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Peso Curso", style={"fontWeight": "600", "fontSize": "0.83rem"}),
                                                        dcc.Input(
                                                            id="input-peso-curso",
                                                            type="number",
                                                            value=config.peso_curso,
                                                            step=0.05,
                                                            min=0.0,
                                                            max=10.0,
                                                            className="form-control form-control-sm",
                                                        ),
                                                    ],
                                                    md=2,
                                                    className="mb-2",
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Pen. Período", style={"fontWeight": "600", "fontSize": "0.83rem"}),
                                                        dcc.Input(
                                                            id="input-pen-periodo",
                                                            type="number",
                                                            value=config.penalidade_periodo,
                                                            step=0.1,
                                                            min=0.0,
                                                            max=20.0,
                                                            className="form-control form-control-sm",
                                                        ),
                                                    ],
                                                    md=2,
                                                    className="mb-2",
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Pen. Instituição", style={"fontWeight": "600", "fontSize": "0.83rem"}),
                                                        dcc.Input(
                                                            id="input-pen-instituicao",
                                                            type="number",
                                                            value=config.penalidade_instituicao,
                                                            step=0.1,
                                                            min=0.0,
                                                            max=20.0,
                                                            className="form-control form-control-sm",
                                                        ),
                                                    ],
                                                    md=2,
                                                    className="mb-2",
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Pen. Curso / Carga", style={"fontWeight": "600", "fontSize": "0.83rem"}),
                                                        dcc.Input(
                                                            id="input-pen-curso-carga",
                                                            type="number",
                                                            value=config.penalidade_curso_carga,
                                                            step=0.1,
                                                            min=0.0,
                                                            max=20.0,
                                                            className="form-control form-control-sm",
                                                        ),
                                                    ],
                                                    md=2,
                                                    className="mb-2",
                                                ),
                                            ]
                                        )
                                    ]
                                ),
                                color="light",
                                style={"marginTop": "10px", "borderRadius": "8px"},
                            ),
                            id="collapse-personalizacao",
                            is_open=False,
                        ),
                    ]
                ),
                style={"borderRadius": "10px", "boxShadow": "0 2px 8px rgba(0,0,0,0.06)", "marginBottom": "16px"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(dbc.CardBody(dcc.Graph(id="grafico-pareamento"))),
                        lg=12,
                        className="mb-3",
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5(
                                        "Síntese da Rede e Afinidade",
                                        style={"fontWeight": "700", "color": "#4a148c", "marginBottom": "10px"},
                                    ),
                                    html.Div(id="resumo-textual-complementar"),
                                ]
                            )
                        ),
                        lg=12,
                        className="mb-3",
                    ),
                ]
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.H5("Métricas", style={"fontWeight": "700", "color": "#4a148c", "marginBottom": "6px"}),
                                    md=4,
                                    className="mb-2",
                                ),
                                dbc.Col(
                                    dbc.DropdownMenu(
                                        label="Métricas por Instituições",
                                        color="light",
                                        toggle_style={
                                            "width": "100%",
                                            "textAlign": "left",
                                            "border": "1px solid #ced4da",
                                            "color": "#212529",
                                            "fontSize": "0.85rem",
                                        },
                                        children=[
                                            dbc.DropdownMenuItem(
                                                dbc.Checklist(
                                                    id="check-metricas-instituicoes",
                                                    options=opcoes_filtro_geral_inst,
                                                    value=["TODAS"],
                                                ),
                                                toggle=False,
                                            )
                                        ],
                                    ),
                                    md=4,
                                    className="mb-2",
                                ),
                                dbc.Col(
                                    dbc.Checklist(
                                        id="flag-exibir-metricas-acima-csv",
                                        options=[{"label": " Exibir métricas", "value": "exibir_acima_csv"}],
                                        value=["exibir_acima_csv"],
                                        style={"fontWeight": "600", "fontSize": "0.84rem", "paddingTop": "6px"},
                                    ),
                                    md=4,
                                    className="mb-2",
                                ),
                            ],
                            className="mb-2 align-items-center",
                        ),
                        html.Div(
                            id="container-tabela-metricas",
                            children=[
                                dash_table.DataTable(
                                    id="tabela-metricas",
                                    columns=[
                                        {"name": "Escopo", "id": "Escopo"},
                                        {"name": "Nós Conectados", "id": "Nós Conectados"},
                                        {"name": "Cobertura V (%)", "id": "Cobertura V (%)"},
                                        {"name": "Pares (|E|)", "id": "Pares (|E|)"},
                                        {"name": "Grau Médio (<k>)", "id": "Grau Médio (<k>)"},
                                        {"name": "Grau Médio U", "id": "Grau Médio U"},
                                        {"name": "DP Grau U", "id": "DP Grau U"},
                                        {"name": "Densidade Bipartida", "id": "Densidade Bipartida"},
                                        {"name": "Hub", "id": "Hub"},
                                        {"name": "Componentes", "id": "Componentes"},
                                        {"name": "Afinidade Média", "id": "Afinidade Média"},
                                        {"name": "Período (%)", "id": "Período (%)"},
                                        {"name": "Instituição (%)", "id": "Instituição (%)"},
                                        {"name": "Curso (%)", "id": "Curso (%)"},
                                    ],
                                    style_table={"overflowX": "auto"},
                                    style_header={"backgroundColor": "#4a148c", "color": "white", "fontWeight": "bold"},
                                    style_cell={"textAlign": "left", "padding": "8px", "fontSize": "0.84rem"},
                                ),
                            ],
                        ),
                    ]
                ),
                style={"borderRadius": "10px", "boxShadow": "0 2px 8px rgba(0,0,0,0.06)", "marginBottom": "14px"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Button(
                                "Explicação das Métricas",
                                id="btn-toggle-explicacao-metricas",
                                color="secondary",
                                outline=True,
                                className="me-2 mb-2",
                            ),
                            dbc.Button(
                                "Explicação dos Algoritmos",
                                id="btn-toggle-explicacao",
                                color="secondary",
                                outline=True,
                                className="me-2 mb-2",
                            ),
                            dbc.Button(
                                "Características da Rede",
                                id="btn-toggle-caracteristicas",
                                color="dark",
                                outline=True,
                                className="me-2 mb-2",
                            ),
                            dbc.Button(
                                "Sugestões de Filtros",
                                id="btn-toggle-sugestoes",
                                color="info",
                                outline=True,
                                className="me-2 mb-2",
                            ),
                            dbc.Button(
                                "Tabela de Pares",
                                id="btn-toggle-pares",
                                color="primary",
                                outline=True,
                                className="mb-2",
                            ),
                        ],
                        xs=12,
                    )
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Collapse(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H6(
                                            "Explicação das Métricas",
                                            style={"fontWeight": "700", "color": "#4a148c", "marginBottom": "8px"},
                                        ),
                                        html.P(
                                            [
                                                html.Strong("Nós Conectados: "),
                                                "quantidade total de participantes ativos na rede formada, separando mentoras (U) e mentoradas recebidas (V).",
                                            ],
                                            style={"fontSize": "0.84rem", "marginBottom": "5px"},
                                        ),
                                        html.P(
                                            [
                                                html.Strong("Pares (|E|): "),
                                                "número total de arestas/vínculos de mentoria formados no recorte.",
                                            ],
                                            style={"fontSize": "0.84rem", "marginBottom": "5px"},
                                        ),
                                        html.P(
                                            [
                                                html.Strong("Cobertura V (%): "),
                                                "percentual de participantes do conjunto V conectadas em relação ao total elegível do recorte.",
                                            ],
                                            style={"fontSize": "0.84rem", "marginBottom": "5px"},
                                        ),
                                        html.P(
                                            [
                                                html.Strong("Grau Médio (<k>), Grau Médio U e DP Grau U: "),
                                                "média de conexões por nó ativo, média de mentoradas por mentora ativa e desvio padrão do grau em U (mede o equilíbrio/dispersão de carga entre mentoras).",
                                            ],
                                            style={"fontSize": "0.84rem", "marginBottom": "5px"},
                                        ),
                                        html.P(
                                            [
                                                html.Strong("Densidade Bipartida e Componentes: "),
                                                "densidade é |E| dividido por (|U| × |V|); componentes indicam quantos subgrupos conexos existem.",
                                            ],
                                            style={"fontSize": "0.84rem", "marginBottom": "5px"},
                                        ),
                                        html.P(
                                            [
                                                html.Strong("Hub e Afinidade Média: "),
                                                "hub é a mentora com mais conexões (k); afinidade média resume o escore [0 a 1] de compatibilidade.",
                                            ],
                                            style={"fontSize": "0.84rem", "marginBottom": "5px"},
                                        ),
                                        html.P(
                                            [
                                                html.Strong("Período (%), Instituição (%) e Curso (%): "),
                                                "percentual de pares que respeitam ingresso anterior da mentora, mesma instituição e mesmo curso.",
                                            ],
                                            style={"fontSize": "0.84rem", "marginBottom": "0"},
                                        ),
                                    ]
                                ),
                                style={"borderRadius": "10px", "boxShadow": "0 2px 8px rgba(0,0,0,0.06)", "height": "100%"},
                            ),
                            id="collapse-explicacao-metricas",
                            is_open=False,
                        ),
                        lg=12,
                        className="mb-3",
                    ),
                    dbc.Col(
                        dbc.Collapse(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H6(
                                            "Explicação dos Algoritmos",
                                            style={"fontWeight": "700", "color": "#4a148c", "marginBottom": "8px"},
                                        ),
                                        html.P(
                                            [
                                                html.Strong("1. Rede Aleatória (Ponto de partida / Baseline): "),
                                                "forma duplas por sorteio respeitando as regras ligadas e o limite de conexões. ",
                                                html.Em("Para que serve: "),
                                                "mostra como seria a rede sem otimização inteligente, servindo de referência para medir os ganhos reais.",
                                            ],
                                            style={"fontSize": "0.84rem", "marginBottom": "7px"},
                                        ),
                                        html.P(
                                            [
                                                html.Strong("2. Gulosa por Afinidade e Carga (Heurística Construtiva Determinística): "),
                                                "ordena candidatos priorizando maior afinidade temática, aderência às regras ativas e menor carga atual da mentora. ",
                                                html.Em("Para que serve: "),
                                                "fornece uma referência determinística rápida e interpretável de alocação gulosa.",
                                            ],
                                            style={"fontSize": "0.84rem", "marginBottom": "7px"},
                                        ),
                                        html.P(
                                            [
                                                html.Strong("3. Matching de Cardinalidade Máxima e Maior Peso (NetworkX): "),
                                                "representa as capacidades das mentoras como slots em um grafo bipartido e prioriza o maior número de pares; entre essas soluções, maximiza o peso total. ",
                                                html.Em("Para que serve: "),
                                                "oferece uma solução estrutural de referência em teoria dos grafos bipartidos para comparar com meta-heurísticas.",
                                            ],
                                            style={"fontSize": "0.84rem", "marginBottom": "7px"},
                                        ),
                                        html.P(
                                            [
                                                html.Strong("4. AG Clássico (Algoritmo Genético Mono-objetivo): "),
                                                "funciona como um ciclo de tentativa, seleção e mistura: cria várias propostas de pareamento, escolhe as melhores e combina partes delas nas próximas rodadas usando uma nota única. ",
                                                html.Em("Para que serve: "),
                                                "encontra boas soluções de forma rápida quando queremos resumir tudo em uma única pontuação.",
                                            ],
                                            style={"fontSize": "0.84rem", "marginBottom": "7px"},
                                        ),
                                        html.P(
                                            [
                                                html.Strong("5. AG Elitista (Com preservação dos campeões): "),
                                                "segue a mesma lógica do AG clássico, mas garante que os melhores pareamentos de uma rodada sejam preservados sem serem perdidos no sorteio ou na mutação. ",
                                                html.Em("Para que serve: "),
                                                "deixa o resultado mais estável quando há muitas regras ativas.",
                                            ],
                                            style={"fontSize": "0.84rem", "marginBottom": "7px"},
                                        ),
                                        html.P(
                                            [
                                                html.Strong("6. NSGA-II (Equilíbrio entre múltiplos objetivos): "),
                                                "em vez de forçar apenas uma nota única, compara soluções equilibrando dois lados ao mesmo tempo: aumentar a afinidade entre mentora e aluna e reduzir quebras de regras ou sobrecarga. ",
                                                html.Em("Para que serve: "),
                                                "é o mais indicado para equilibrar afinidade, regras acadêmicas e limite de alunas por mentora.",
                                            ],
                                            style={"fontSize": "0.84rem", "marginBottom": "0"},
                                        ),
                                    ]
                                ),
                                style={"borderRadius": "10px", "boxShadow": "0 2px 8px rgba(0,0,0,0.06)", "height": "100%"},
                            ),
                            id="collapse-explicacao",
                            is_open=False,
                        ),
                        lg=12,
                        className="mb-3",
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Collapse(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5("Características da Rede", style={"fontWeight": "700", "color": "#4a148c", "marginBottom": "10px"}),
                                        dash_table.DataTable(
                                            id="tabela-caracteristicas-rede",
                                            columns=[
                                                {"name": "Instituição", "id": "Instituição"},
                                                {"name": "Curso", "id": "Curso"},
                                                {"name": "Turno", "id": "Turno"},
                                                {"name": "Total Participantes", "id": "Total Participantes"},
                                                {"name": "Qtd. Mentoradas", "id": "Qtd. Mentoradas"},
                                                {"name": "Qtd. Mentoras", "id": "Qtd. Mentoras"},
                                                {"name": "Qtd. Embaixadoras", "id": "Qtd. Embaixadoras"},
                                                {"name": "Distribuição por Papel", "id": "Distribuição por Papel"},
                                                {"name": "Semestre de Ingresso", "id": "Semestre de Ingresso"},
                                                {"name": "Principais Áreas", "id": "Principais Áreas"},
                                            ],
                                            page_size=15,
                                            sort_action="native",
                                            filter_action="native",
                                            style_table={"overflowX": "auto"},
                                            style_header={"backgroundColor": "#4a148c", "color": "white", "fontWeight": "bold"},
                                            style_cell={"textAlign": "left", "padding": "8px", "fontSize": "0.84rem"},
                                        ),
                                    ]
                                ),
                                style={"borderRadius": "10px", "boxShadow": "0 2px 8px rgba(0,0,0,0.06)", "marginBottom": "16px"},
                            ),
                            id="collapse-caracteristicas",
                            is_open=False,
                        ),
                        lg=12,
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Collapse(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5("Sugestões de Filtros por Tipo de Análise", style={"fontWeight": "700", "color": "#4a148c", "marginBottom": "12px"}),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.H6("1. Pareamento Ideal com Equilíbrio de Carga", style={"fontWeight": "700", "color": "#202124"}),
                                                                html.P("Objetivo: maximizar afinidade e cumprir regras sem sobrecarregar mentoras.", style={"fontSize": "0.84rem", "marginBottom": "4px"}),
                                                                html.Ul(
                                                                    [
                                                                        html.Li("Instituição: Todas as Instituições"),
                                                                        html.Li("Algoritmo: NSGA-II (Multiobjetivo)"),
                                                                        html.Li("Papéis: Mentora + Mentorada"),
                                                                        html.Li("Regras: Período, Instituição e Curso marcadas"),
                                                                        html.Li("Limite de conexões por par: 2"),
                                                                    ],
                                                                    style={"fontSize": "0.83rem", "marginBottom": "0"},
                                                                ),
                                                            ]
                                                        ),
                                                        color="light",
                                                        outline=True,
                                                        className="mb-3",
                                                    ),
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.H6("2. Diagnóstico por Instituição Específica", style={"fontWeight": "700", "color": "#202124"}),
                                                                html.P("Objetivo: avaliar aderência local, cursos e capacidade interna de uma IES.", style={"fontSize": "0.84rem", "marginBottom": "4px"}),
                                                                html.Ul(
                                                                    [
                                                                        html.Li("Instituição: marcar a IES desejada (ex.: UPE)"),
                                                                        html.Li("Algoritmo: NSGA-II ou AG Elitista"),
                                                                        html.Li("Papéis: Mentora + Mentorada"),
                                                                        html.Li("Regras: Período, Instituição e Curso marcadas"),
                                                                        html.Li("Limite de conexões por par: 2 ou 3"),
                                                                    ],
                                                                    style={"fontSize": "0.83rem", "marginBottom": "0"},
                                                                ),
                                                            ]
                                                        ),
                                                        color="light",
                                                        outline=True,
                                                        className="mb-3",
                                                    ),
                                                    md=6,
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.H6("3. Simulação de Escassez / Flexibilização de Regras", style={"fontWeight": "700", "color": "#202124"}),
                                                                html.P("Objetivo: testar o impacto quando há poucas mentoras do mesmo curso ou instituição.", style={"fontSize": "0.84rem", "marginBottom": "4px"}),
                                                                html.Ul(
                                                                    [
                                                                        html.Li("Instituição: Todas as Instituições"),
                                                                        html.Li("Algoritmo: NSGA-II (Multiobjetivo)"),
                                                                        html.Li("Papéis: Mentora + Mentorada (+ Embaixadora se necessário)"),
                                                                        html.Li("Regras: manter Período e desmarcar Curso ou Instituição"),
                                                                        html.Li("Limite de conexões por par: 3 ou 4"),
                                                                    ],
                                                                    style={"fontSize": "0.83rem", "marginBottom": "0"},
                                                                ),
                                                            ]
                                                        ),
                                                        color="light",
                                                        outline=True,
                                                    ),
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.H6("4. Benchmark Contra Sorteio (Baseline)", style={"fontWeight": "700", "color": "#202124"}),
                                                                html.P("Objetivo: mensurar o ganho do algoritmo otimizado frente a uma alocação aleatória.", style={"fontSize": "0.84rem", "marginBottom": "4px"}),
                                                                html.Ul(
                                                                    [
                                                                        html.Li("Instituição: Todas as Instituições"),
                                                                        html.Li("Algoritmo: alternar entre Rede Aleatória e NSGA-II"),
                                                                        html.Li("Papéis: Mentora + Mentorada"),
                                                                        html.Li("Regras: Período, Instituição e Curso marcadas"),
                                                                        html.Li("Limite de conexões por par: 2"),
                                                                    ],
                                                                    style={"fontSize": "0.83rem", "marginBottom": "0"},
                                                                ),
                                                            ]
                                                        ),
                                                        color="light",
                                                        outline=True,
                                                    ),
                                                    md=6,
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                                style={"borderRadius": "10px", "boxShadow": "0 2px 8px rgba(0,0,0,0.06)", "marginBottom": "16px"},
                            ),
                            id="collapse-sugestoes",
                            is_open=False,
                        ),
                        lg=12,
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Collapse(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(html.H5("Pares Formados", style={"fontWeight": "700", "color": "#4a148c"}), md=8),
                                                dbc.Col(
                                                    dbc.Button("Exportar CSV", id="btn-exportar-csv", color="success", size="sm", className="float-end"),
                                                    md=4,
                                                ),
                                            ],
                                            className="mb-2",
                                        ),
                                        dcc.Download(id="download-csv"),
                                        dash_table.DataTable(
                                            id="tabela-pares",
                                            columns=[
                                                {"name": "Mentora (U)", "id": "Mentora (U)"},
                                                {"name": "Inst. Mentora", "id": "Inst. Mentora"},
                                                {"name": "Curso Mentora", "id": "Curso Mentora"},
                                                {"name": "Ingresso U", "id": "Ingresso U"},
                                                {"name": "Participante (V)", "id": "Participante (V)"},
                                                {"name": "Papel V", "id": "Papel V"},
                                                {"name": "Inst. V", "id": "Inst. V"},
                                                {"name": "Curso V", "id": "Curso V"},
                                                {"name": "Ingresso V", "id": "Ingresso V"},
                                                {"name": "Afinidade", "id": "Afinidade"},
                                                {"name": "Status", "id": "Status"},
                                            ],
                                            page_size=15,
                                            sort_action="native",
                                            filter_action="native",
                                            style_table={"overflowX": "auto"},
                                            style_header={"backgroundColor": "#4a148c", "color": "white", "fontWeight": "bold"},
                                            style_cell={"textAlign": "left", "padding": "8px", "fontSize": "0.85rem"},
                                        ),
                                    ]
                                ),
                                style={"borderRadius": "10px", "boxShadow": "0 2px 8px rgba(0,0,0,0.06)", "marginBottom": "24px"},
                            ),
                            id="collapse-pares",
                            is_open=False,
                        ),
                        lg=12,
                    )
                ]
            ),
        ],
        fluid=True,
        style={"paddingLeft": "0", "paddingRight": "0"},
    )
