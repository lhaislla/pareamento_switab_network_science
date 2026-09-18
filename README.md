# Otimização de Pareamento em Redes de Mentoria: Um Estudo no SWITAB

Repositório dedicado ao desenvolvimento do projeto de pesquisa voltado à modelagem e otimização do pareamento de mentoras e mentoradas no programa SWITAB, utilizando conceitos de Ciência das Redes e Teoria de Grafos.

## 📊 Apresentação (Slides)
Você pode acompanhar a estruturação e os slides da pesquisa através do link oficial no Canva:
* **[Acessar Apresentação no Canva](https://canva.link/gdvcyr9c82roieo)**

## 🎯 Sobre o Projeto

O objetivo deste projeto é modelar o ecossistema de mentoria do SWITAB como um grafo bipartido e aplicar algoritmos de otimização para realizar o pareamento entre mentoras e mentoradas. O modelo busca respeitar restrições estruturais (como mesma instituição, mesmo curso e precedência de período) enquanto maximiza a afinidade, a coesão social e a retenção de estudantes em cursos de tecnologia.

### Pergunta de Pesquisa Principal
> **Como modelar e otimizar a estratégia de pareamento entre mentoras e mentoradas de forma a maximizar a permanência, a coesão da rede e o sentimento de pertencimento de estudantes de tecnologia no SWITAB?**

## 🛠️ Tecnologias e Bibliotecas Utilizadas

O projeto é desenvolvido em **Python**, utilizando as seguintes bibliotecas principais:

* **[Python 3.x](https://www.python.org/):** Linguagem principal de programação.
* **[Pandas](https://pandas.pydata.org/):** Manipulação, limpeza e estruturação dos dados dos formulários do SWITAB.
* **[NetworkX](https://networkx.org/):** Construção, manipulação e análise topológica das redes (grafos bipartidos, métricas de centralidade e coeficiente de aglomeração).
* **[SciPy](https://scipy.org/):** Algoritmos de otimização e resolução de problemas de pareamento ponderado (*Maximum Weight Bipartite Matching*).
* **[Matplotlib](https://matplotlib.org/) / [Seaborn](https://seaborn.pydata.org/):** Visualização gráfica das redes e distribuições estatísticas.

## 📂 Estrutura do Repositório

```text
├── data/               # Bases de dados e planilhas anonimizadas (não versionadas)
├── notebooks/          # Jupyter Notebooks com as etapas de análise e algoritmos
├── src/                # Códigos-fonte em Python
├── README.md           # Documentação do projeto
└── requirements.txt    # Dependências do projeto

## ⚙️ Tutorial de Instalação e Execução

Siga os passos abaixo para configurar o ambiente de desenvolvimento na sua máquina utilizando o terminal do VS Code.

### 1. Clonar o Repositório

Abra o terminal e clone o projeto:

```bash
git clone https://github.com/lhaislla/pareamento_switab_network_science
cd pareamento_switab_network_science
```

### 2. Criar o Ambiente Virtual (`venv`)

O ambiente virtual isola as dependências do projeto. Execute o comando correspondente ao seu sistema operacional na raiz do repositório.

**Windows (Prompt de Comando ou PowerShell):**

```bash
python -m venv venv
```

**Linux / macOS:**

```bash
python3 -m venv venv
```

### 3. Ativar o Ambiente Virtual

Para ativar o ambiente criado, utilize o comando correspondente ao seu sistema operacional.

**Windows (PowerShell):**

```powershell
.\venv\Scripts\Activate.ps1
```

> **Nota:** Se o PowerShell bloquear a execução do script, execute o comando abaixo antes:

```powershell
Set-ExecutionPolicy Unrestricted -Scope Process
```

**Windows (Command Prompt - CMD):**

```cmd
venv\Scripts\activate.bat
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

Você saberá que a ativação foi realizada corretamente quando o nome `(venv)` aparecer no início da linha do terminal.

### 4. Instalar as Dependências

Com o ambiente virtual ativado, instale todas as bibliotecas necessárias listadas no arquivo `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Como Executar o Projeto

Para abrir e testar os códigos de análise interativa, inicie o Jupyter Notebook:

```bash
jupyter notebook
```

Em seguida, navegue até a pasta `src/` e abra os arquivos de rotina criados para o projeto.
