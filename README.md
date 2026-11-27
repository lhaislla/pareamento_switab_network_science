# Pareamento SWITAB: Dashboard de Mentoria e Comunidades

Aplicação em Python para explorar o pareamento entre mentoras e mentoradas e analisar redes de conexão do programa SWITAB. O dashboard permite comparar estratégias de pareamento, aplicar filtros e visualizar métricas e comunidades.

## Apresentações

- [Apresentação da 1ª entrega](https://canva.link/gdvcyr9c82roieo)
- [Apresentação da 2ª entrega](https://canva.link/yzhk69u9ns8glf9)
- [Apresentação da 3ª entrega](https://canva.link/2eqjnywxgcnxty0)

## Sobre o projeto

O painel de pareamento representa as relações como um grafo bipartido e permite considerar afinidade temática, instituição, curso, período de ingresso e limite de alunas por mentora. Estão disponíveis estratégias aleatória, gulosa, matching de cardinalidade máxima com maior afinidade, algoritmos genéticos e NSGA-II.

O painel de comunidades constrói uma rede de conexões ponderada e permite comparar Louvain, Label Propagation assíncrona ponderada, Greedy Modularity e Girvan-Newman, além de consultar métricas como modularidade, densidade, grau e proporção de conexões entre participantes do mesmo turno.

O projeto é uma ferramenta inicial de apoio à exploração e à análise do SWITAB. As simulações e métricas, por si só, não demonstram retenção, pertencimento nem impacto causal do programa.

### Objetivo da ferramenta

> Como representar as conexões entre participantes e explorar estratégias de pareamento e estrutura de comunidades no SWITAB?

### Interpretação dos algoritmos e métricas

- O matching prioriza a maior quantidade possível de pares e, entre soluções com essa cardinalidade, maximiza a soma dos pesos (afinidade com penalidade para regras violadas). Restrições desativadas ou a opção de forçar cobertura podem permitir pares que violem regras; confira os indicadores de validade da alocação.
- A rede de conexões é construída como dirigida e ponderada. A detecção de comunidades usa uma projeção não direcionada que soma os pesos de conexões recíprocas.
- Os pesos representam força de conexão. Nas métricas de caminhos mínimos, inclusive betweenness e a remoção de arestas do Girvan-Newman, a distância é calculada como o inverso da força. Louvain, Label Propagation e Greedy Modularity usam os pesos como força.
- “Conexões no Mesmo Turno” é a porcentagem de arestas direcionadas cujas duas pontas têm turno conhecido e igual. É uma estatística descritiva, não um coeficiente de homofilia ajustado pela distribuição dos turnos.
- `data/base_conexoes_switab_ficticia.csv` é uma base fictícia de demonstração. Não use resultados dessa base para afirmar propriedades empíricas do SWITAB; configure uma base real, autorizada e documentada para análises do programa.

## Tecnologias

- **Python 3.10 ou superior**
- **Dash** e **Dash Bootstrap Components** para a aplicação web
- **Plotly** para visualizações interativas
- **Pandas** e **NumPy** para manipulação e cálculo de dados
- **NetworkX** para construção e análise de grafos
- **python-dotenv** para carregar configurações locais do arquivo `.env`

As dependências estão listadas em [`requirements.txt`](requirements.txt).

## Estrutura do projeto

```text
.
├── app.py                         # Inicialização do dashboard e callbacks
├── config/
│   └── settings.py                # Configurações e variáveis de ambiente
├── core/
│   ├── pareamento_engine.py       # Ingestão, pareamento e métricas
│   └── comunidades_engine.py      # Grafos e detecção de comunidades
├── components/
│   └── cards.py                   # Componentes visuais reutilizáveis
├── pages/
│   ├── pareamento_page.py         # Layout do painel de pareamento
│   └── comunidades_page.py        # Layout do painel de comunidades
├── viz/
│   ├── graficos_pareamento.py     # Gráficos de pareamento
│   └── graficos_comunidades.py    # Gráficos de comunidades
├── data/
│   ├── SWITAB.csv                 # Base do pareamento
│   └── base_conexoes_switab_ficticia.csv
├── .env.example                   # Modelo de configuração local
├── requirements.txt               # Dependências Python
└── README.md
```

O arquivo `dashboard_switab_pareamento_comunidades.py` contém uma implementação monolítica/alternativa. O ponto de entrada documentado para a estrutura modular é `app.py`.

## Instalação e execução

### 1. Clonar o repositório

```powershell
git clone https://github.com/lhaislla/pareamento_switab_network_science.git
cd pareamento_switab_network_science
```

### 2. Criar e ativar o ambiente virtual

No Windows, usando PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Se o PowerShell impedir a ativação, permita scripts locais somente nesta sessão:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

No macOS ou Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar as dependências

Com o ambiente virtual ativado:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configurar o ambiente

O arquivo `.env` é opcional. Para criar uma cópia local do modelo:

```powershell
Copy-Item .env.example .env
```

Não versionar o `.env` caso contenha configurações ou dados locais. Sem esse arquivo, a aplicação utiliza os valores padrão definidos em `config/settings.py`.



Os caminhos relativos das bases são resolvidos a partir da raiz do projeto. O dashboard exige os arquivos indicados: se uma base estiver ausente, vazia ou inválida, a inicialização informa o erro em vez de substituí-la por dados sintéticos.

### 5. Iniciar o dashboard

Na raiz do repositório, com o ambiente virtual ativado:

```powershell
python .\app.py
```

Abra no navegador o endereço exibido no terminal; com a configuração padrão, é <http://127.0.0.1:8050>.
