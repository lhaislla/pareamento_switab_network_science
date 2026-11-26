import hashlib
import io
import math
import os
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

import dash
import dash_bootstrap_components as dbc
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html




COLUNA_ALIASES: Dict[str, List[str]] = {
    "id_anonimo": ["id_anonimo", "id", "codigo", "participante_id", "matricula_anonima"],
    "papel": ["papel", "pape_interesse", "papel_interesse", "funcao", "tipo_participante", "perfil"],
    "instituicao": ["instituicao", "ies", "universidade", "instituicao_ensino", "orgao"],
    "curso": ["curso", "graduacao", "programa", "curso_graduacao"],
    "turno": ["turno", "periodo_turno", "horario"],
    "ano_ingresso": ["ano_ingresso", "semestre_ingresso", "periodo_ingresso", "ingresso", "ano_semestre"],
    "areas_interesse": [
        "areas_interesse",
        "area_interesse",
        "areas",
        "interesses",
        "temas_interesse",
        "areas_de_interesse",
    ],
}

COLUNAS_OBRIGATORIAS = ["id_anonimo", "papel", "instituicao", "curso", "ano_ingresso"]


@dataclass
class RelatorioIngestao:
    origem: str
    total_linhas_brutas: int = 0
    total_linhas_validas: int = 0
    linhas_descartadas: int = 0
    colunas_mapeadas: Dict[str, str] = field(default_factory=dict)
    avisos: List[str] = field(default_factory=list)


def _normalizar_nome_coluna(col: str) -> str:
    """Normaliza o nome de uma coluna para facilitar o mapeamento de aliases."""
    sem_acento = "".join(
        ch for ch in unicodedata.normalize("NFKD", str(col).strip().lower())
        if not unicodedata.combining(ch)
    )
    return re.sub(r"[^a-z0-9]+", "_", sem_acento).strip("_")


def _normalizar_texto(valor: Any, padrao: str = "Não Informado") -> str:
    """Limpa um valor textual e substitui valores ausentes pelo padrão informado."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return padrao
    texto = re.sub(r"\s+", " ", str(valor).strip())
    if not texto or texto.lower() in {"nan", "none", "null", "n/a", "na"}:
        return padrao
    return texto


def _normalizar_chave_comparacao(valor: str) -> str:
    """Padroniza texto para comparações sem distinção de caixa ou acentuação."""
    sem_acento = "".join(
        ch for ch in unicodedata.normalize("NFKD", str(valor).strip().lower())
        if not unicodedata.combining(ch)
    )
    return re.sub(r"\s+", " ", sem_acento)


def _parse_periodo_ingresso(valor: Any, padrao: float = 2026.1) -> float:
    """Converte o período de ingresso para o formato numérico ano.semestre."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return padrao
    if isinstance(valor, (int, float, np.number)):
        num = float(valor)
        if np.isfinite(num) and 1980.0 <= num <= 2100.0:
            return round(num, 1)
        return padrao

    texto = str(valor).strip().replace(",", ".")
    match_semestre = re.match(r"^(\d{4})[./\-_: ]?([12])$", texto)
    if match_semestre:
        ano = int(match_semestre.group(1))
        sem = int(match_semestre.group(2))
        if 1980 <= ano <= 2100:
            return float(f"{ano}.{sem}")

    num = pd.to_numeric(texto, errors="coerce")
    if pd.notna(num) and np.isfinite(float(num)) and 1980.0 <= float(num) <= 2100.0:
        return round(float(num), 1)
    return padrao


def _extrair_conjunto_areas(valor: Any) -> set:
    """Extrai e normaliza as áreas de interesse informadas em uma string."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return set()
    texto = str(valor).strip()
    if not texto or texto.lower() in {"nan", "none", "null", "não informado", "nao informado"}:
        return set()
    partes = re.split(r"[,;|/]+", texto)
    areas = set()
    for parte in partes:
        limpa = _normalizar_chave_comparacao(parte)
        if limpa and limpa not in {"nan", "none", "null", "nao informado"}:
            areas.add(limpa)
    return areas


def _validar_e_limpar_dataframe(df_bruto: pd.DataFrame, origem: str) -> Tuple[pd.DataFrame, RelatorioIngestao]:
    """Padroniza colunas e registros da base e retorna o relatório de ingestão."""
    rel = RelatorioIngestao(origem=origem, total_linhas_brutas=len(df_bruto))
    if df_bruto is None or len(df_bruto) == 0:
        rel.avisos.append("Base vazia recebida na ingestão.")
        return pd.DataFrame(columns=COLUNAS_OBRIGATORIAS + ["turno", "areas_interesse"]), rel

    mapa_orig_por_norm = {_normalizar_nome_coluna(c): c for c in df_bruto.columns}
    df_pad = pd.DataFrame(index=df_bruto.index)

    for canonico, aliases in COLUNA_ALIASES.items():
        origem_col = None
        for alias in aliases:
            alias_norm = _normalizar_nome_coluna(alias)
            if alias_norm in mapa_orig_por_norm:
                origem_col = mapa_orig_por_norm[alias_norm]
                break
        if origem_col is not None:
            df_pad[canonico] = df_bruto[origem_col]
            rel.colunas_mapeadas[canonico] = origem_col
        else:
            if canonico in COLUNAS_OBRIGATORIAS:
                rel.avisos.append(f"Coluna obrigatória ausente ('{canonico}'); aplicado valor padrão seguro.")
            if canonico == "ano_ingresso":
                df_pad[canonico] = 2026.1
            elif canonico == "turno":
                df_pad[canonico] = "Não Informado"
            elif canonico == "areas_interesse":
                df_pad[canonico] = "Não Informado"
            else:
                df_pad[canonico] = "Não Informado"

    df_pad["id_anonimo"] = df_pad["id_anonimo"].apply(lambda v: _normalizar_texto(v, padrao=""))
    df_pad["papel"] = df_pad["papel"].apply(lambda v: _normalizar_texto(v, padrao="Não Informado"))
    df_pad["instituicao"] = df_pad["instituicao"].apply(lambda v: _normalizar_texto(v, padrao="Não Informada"))
    df_pad["curso"] = df_pad["curso"].apply(lambda v: _normalizar_texto(v, padrao="Não Informado"))
    df_pad["turno"] = df_pad["turno"].apply(lambda v: _normalizar_texto(v, padrao="Não Informado"))
    df_pad["areas_interesse"] = df_pad["areas_interesse"].apply(lambda v: _normalizar_texto(v, padrao="Não Informado"))
    df_pad["ano_ingresso"] = df_pad["ano_ingresso"].apply(_parse_periodo_ingresso)

    sem_id_mask = df_pad["id_anonimo"] == ""
    if sem_id_mask.any():
        qtd_sem_id = int(sem_id_mask.sum())
        df_pad.loc[sem_id_mask, "id_anonimo"] = [f"ID_AUTO_{idx + 1:04d}" for idx in range(qtd_sem_id)]
        rel.avisos.append(f"{qtd_sem_id} registro(s) sem ID receberam identificador sintético seguro.")

    antes_dup = len(df_pad)
    df_pad = df_pad.drop_duplicates(subset=["id_anonimo"], keep="first").reset_index(drop=True)
    if len(df_pad) < antes_dup:
        dups = antes_dup - len(df_pad)
        rel.linhas_descartadas += dups
        rel.avisos.append(f"{dups} registro(s) duplicado(s) por 'id_anonimo' removido(s).")

    rel.total_linhas_validas = len(df_pad)
    return df_pad, rel


def gerar_dataset_sintetico(seed: int = 42) -> pd.DataFrame:
    """Gera uma base reprodutível de mentoras, mentoradas e embaixadoras."""
    rng = np.random.default_rng(seed)
    cursos_por_instituicao = {
        "UPE": ["Engenharia da Computação", "Sistemas de Informação", "Licenciatura em Computação"],
        "UFRPE": ["Sistemas de Informação", "Ciência da Computação", "Licenciatura em Computação"],
        "UFPE": ["Ciência da Computação", "Engenharia da Computação", "Sistemas de Informação"],
        "UNICAP": ["Ciência da Computação", "Sistemas de Informação", "Engenharia de Software"],
        "CESAR School": ["Ciência da Computação", "Engenharia de Software", "Design"],
    }
    instituicoes = list(cursos_por_instituicao.keys())
    areas_pool = [
        "Ciência de Dados",
        "Inteligência Artificial",
        "Desenvolvimento Web",
        "Engenharia de Software",
        "Segurança da Informação",
        "UX/UI Design",
        "Pesquisa Acadêmica",
        "Redes de Computadores",
        "Gestão de Projetos",
    ]

    registros: List[Dict[str, Any]] = []
    periodos_mentoras = [2021.1, 2021.2, 2022.1, 2022.2, 2023.1, 2023.2, 2024.1, 2024.2]
    for i in range(68):
        inst = str(rng.choice(instituicoes))
        curso = str(rng.choice(cursos_por_instituicao[inst]))
        areas = ", ".join(rng.choice(areas_pool, size=rng.integers(2, 4), replace=False))
        turno = str(rng.choice(["Manhã", "Tarde", "Noite", "Integral"]))
        registros.append(
            {
                "id_anonimo": f"ID_M{i + 1:04d}",
                "papel": "Mentora",
                "instituicao": inst,
                "curso": curso,
                "turno": turno,
                "ano_ingresso": float(rng.choice(periodos_mentoras)),
                "areas_interesse": areas,
            }
        )

    periodos_mentoradas = [2024.1, 2024.2, 2025.1, 2025.2, 2026.1, 2026.2]
    for j in range(92):
        inst = str(rng.choice(instituicoes))
        curso = str(rng.choice(cursos_por_instituicao[inst]))
        areas = ", ".join(rng.choice(areas_pool, size=rng.integers(2, 4), replace=False))
        turno = str(rng.choice(["Manhã", "Tarde", "Noite", "Integral"]))
        registros.append(
            {
                "id_anonimo": f"ID_A{j + 1:04d}",
                "papel": "Mentorada",
                "instituicao": inst,
                "curso": curso,
                "turno": turno,
                "ano_ingresso": float(rng.choice(periodos_mentoradas)),
                "areas_interesse": areas,
            }
        )

    for k in range(18):
        inst = str(rng.choice(instituicoes))
        curso = str(rng.choice(cursos_por_instituicao[inst]))
        areas = ", ".join(rng.choice(areas_pool, size=rng.integers(2, 4), replace=False))
        turno = str(rng.choice(["Manhã", "Tarde", "Noite", "Integral"]))
        registros.append(
            {
                "id_anonimo": f"ID_E{k + 1:04d}",
                "papel": "Embaixadora",
                "instituicao": inst,
                "curso": curso,
                "turno": turno,
                "ano_ingresso": float(rng.choice(periodos_mentoras + periodos_mentoradas)),
                "areas_interesse": areas,
            }
        )

    return pd.DataFrame(registros)


def _ler_arquivo_tabular(path: str) -> pd.DataFrame:
    """Lê uma planilha ou tenta delimitadores comuns para importar um arquivo tabular."""
    ext = os.path.splitext(path)[1].lower()
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    for sep in [",", ";", "\t"]:
        try:
            df_try = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
            if df_try.shape[1] >= 2:
                return df_try
        except Exception:
            continue
    return pd.read_csv(path, encoding="latin-1")


def carregar_dados_base_com_relatorio(
    caminho_arquivo: Optional[str] = None,
) -> Tuple[pd.DataFrame, RelatorioIngestao]:
    """Carrega e valida a base de pareamento sem substituí-la por dados sintéticos."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if caminho_arquivo is None:
        caminho_arquivo = os.path.join(base_dir, "data", "SWITAB.csv")
    elif not os.path.isabs(caminho_arquivo):
        caminho_arquivo = os.path.join(base_dir, caminho_arquivo)

    if not os.path.isfile(caminho_arquivo):
        raise FileNotFoundError(f"Arquivo de pareamento não encontrado: {caminho_arquivo}")

    df_raw = _ler_arquivo_tabular(caminho_arquivo)
    if df_raw.empty:
        raise ValueError(f"Arquivo de pareamento vazio: {caminho_arquivo}")

    df_limpo, rel = _validar_e_limpar_dataframe(df_raw, origem=f"Arquivo: {caminho_arquivo}")
    colunas_ausentes = [coluna for coluna in COLUNAS_OBRIGATORIAS if coluna not in rel.colunas_mapeadas]
    if colunas_ausentes:
        raise ValueError(
            f"Arquivo de pareamento sem colunas obrigatórias reconhecidas "
            f"({', '.join(colunas_ausentes)}): {caminho_arquivo}"
        )
    if df_limpo.empty:
        raise ValueError(f"Arquivo de pareamento não contém registros válidos: {caminho_arquivo}")

    return df_limpo, rel


def construir_caracteristicas_gerais_rede(
    df: pd.DataFrame,
    instituicoes_selecionadas: Optional[List[str]] = None,
    cursos_selecionados: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Resume participantes, papéis, períodos e áreas por instituição e curso."""
    df_base, _ = _validar_e_limpar_dataframe(df, origem="filtro_caracteristicas")

    if instituicoes_selecionadas:
        inst_clean_sel = [str(x).strip() for x in instituicoes_selecionadas if str(x).strip()]
        individuais = [x for x in inst_clean_sel if x != "TODAS"]
        if individuais:
            alvo_inst = {_normalizar_chave_comparacao(x) for x in individuais}
            df_base = df_base[df_base["instituicao"].apply(lambda v: _normalizar_chave_comparacao(v) in alvo_inst)].copy()

    if cursos_selecionados:
        cursos_clean_sel = [str(x).strip() for x in cursos_selecionados if str(x).strip()]
        cursos_ind = [x for x in cursos_clean_sel if x != "TODOS_CURSOS"]
        if cursos_ind:
            alvo_curso = {_normalizar_chave_comparacao(x) for x in cursos_ind}
            df_base = df_base[df_base["curso"].apply(lambda v: _normalizar_chave_comparacao(v) in alvo_curso)].copy()

    if len(df_base) == 0:
        return []

    linhas: List[Dict[str, Any]] = []
    for (inst, curso), grp in df_base.groupby(["instituicao", "curso"], sort=True):
        turnos_str = ", ".join([f"{t} ({c})" for t, c in grp["turno"].value_counts().items()])
        qtd_total = int(len(grp))
        cont_papel = grp["papel"].value_counts().to_dict()
        papeis_str = ", ".join([f"{p}: {c}" for p, c in cont_papel.items()])
        qtd_mentoradas = int(grp[grp["papel"].str.lower() == "mentorada"].shape[0])
        qtd_mentoras = int(grp[grp["papel"].str.lower() == "mentora"].shape[0])
        qtd_embaix = int(grp[grp["papel"].str.lower() == "embaixadora"].shape[0])
        periodos_ord = sorted(grp["ano_ingresso"].unique())
        semestres_str = (
            f"{periodos_ord[0]} a {periodos_ord[-1]} ({len(periodos_ord)} sem.)"
            if len(periodos_ord) > 1
            else f"{periodos_ord[0]}"
        )

        areas_cont: Dict[str, int] = {}
        for txt in grp["areas_interesse"].fillna(""):
            for item in _extrair_conjunto_areas(txt):
                areas_cont[item] = areas_cont.get(item, 0) + 1
        top_areas = ", ".join([k.title() for k, _ in sorted(areas_cont.items(), key=lambda x: (-x[1], x[0]))[:3]])
        if not top_areas:
            top_areas = "Não informado"

        linhas.append(
            {
                "Instituição": inst,
                "Curso": curso,
                "Turno": turnos_str,
                "Total Participantes": qtd_total,
                "Qtd. Mentoradas": qtd_mentoradas,
                "Qtd. Mentoras": qtd_mentoras,
                "Qtd. Embaixadoras": qtd_embaix,
                "Distribuição por Papel": papeis_str,
                "Semestre de Ingresso": semestres_str,
                "Principais Áreas": top_areas,
            }
        )

    return linhas


def preparar_conjuntos_bipartidos(
    df: pd.DataFrame,
    papeis_selecionados: Optional[List[str]] = None,
    instituicoes_selecionadas: Optional[List[str]] = None,
    cursos_selecionados: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filtra a base e separa os conjuntos de mentoras e mentoradas."""
    df_clean, _ = _validar_e_limpar_dataframe(df, origem="conjuntos_bipartidos")

    if instituicoes_selecionadas:
        inst_clean_sel = [str(x).strip() for x in instituicoes_selecionadas if str(x).strip()]
        individuais_marcadas = [x for x in inst_clean_sel if x != "TODAS"]
        if individuais_marcadas:
            alvo_inst = {_normalizar_chave_comparacao(x) for x in individuais_marcadas}
            df_clean = df_clean[
                df_clean["instituicao"].apply(lambda v: _normalizar_chave_comparacao(v) in alvo_inst)
            ].copy()

    if cursos_selecionados:
        cursos_clean_sel = [str(x).strip() for x in cursos_selecionados if str(x).strip()]
        cursos_individuais = [x for x in cursos_clean_sel if x != "TODOS_CURSOS"]
        if cursos_individuais:
            alvo_curso = {_normalizar_chave_comparacao(x) for x in cursos_individuais}
            df_clean = df_clean[
                df_clean["curso"].apply(lambda v: _normalizar_chave_comparacao(v) in alvo_curso)
            ].copy()

    if not papeis_selecionados:
        papeis_selecionados = ["Mentora", "Mentorada"]

    papeis_lower = {_normalizar_chave_comparacao(p) for p in papeis_selecionados}
    df_filtrado_papel = df_clean[df_clean["papel"].apply(lambda p: _normalizar_chave_comparacao(p) in papeis_lower)].copy()

    mentoras = df_filtrado_papel[df_filtrado_papel["papel"].apply(lambda p: _normalizar_chave_comparacao(p) == "mentora")].copy()
    mentoradas = df_filtrado_papel[df_filtrado_papel["papel"].apply(lambda p: _normalizar_chave_comparacao(p) != "mentora")].copy()

    return mentoras.reset_index(drop=True), mentoradas.reset_index(drop=True)


def calcular_afinidade_par(
    m_row: pd.Series,
    a_row: pd.Series,
    col_texto: str = "areas_interesse",
    pesos: Tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> float:
    """Calcula a afinidade entre duas participantes por áreas, instituição e curso."""
    w_jaccard, w_inst, w_curso = pesos
    soma_pesos = w_jaccard + w_inst + w_curso
    if soma_pesos > 0:
        w_jaccard, w_inst, w_curso = w_jaccard / soma_pesos, w_inst / soma_pesos, w_curso / soma_pesos
    else:
        w_jaccard, w_inst, w_curso = 0.70, 0.15, 0.15

    set_m = _extrair_conjunto_areas(m_row.get(col_texto, ""))
    set_a = _extrair_conjunto_areas(a_row.get(col_texto, ""))

    if not set_m and not set_a:
        jaccard = 0.25
    elif not set_m or not set_a:
        jaccard = 0.10
    else:
        uniao = set_m | set_a
        jaccard = (len(set_m & set_a) / len(uniao)) if uniao else 0.0

    bonus_inst = (
        w_inst
        if _normalizar_chave_comparacao(str(m_row.get("instituicao", "")))
        == _normalizar_chave_comparacao(str(a_row.get("instituicao", "")))
        else 0.0
    )
    bonus_curso = (
        w_curso
        if _normalizar_chave_comparacao(str(m_row.get("curso", "")))
        == _normalizar_chave_comparacao(str(a_row.get("curso", "")))
        else 0.0
    )

    score = w_jaccard * jaccard + bonus_inst + bonus_curso
    return float(np.clip(round(score, 4), 0.0, 1.0))


def construir_matriz_compatibilidade(
    mentoras: pd.DataFrame,
    mentoradas: pd.DataFrame,
    pesos_afinidade: Tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> Dict[str, np.ndarray]:
    """Cria matrizes de afinidade e de compatibilidade para cada par possível."""
    n_m, n_a = len(mentoras), len(mentoradas)
    afinidade = np.zeros((n_m, n_a), dtype=float)
    match_inst = np.zeros((n_m, n_a), dtype=bool)
    match_curso = np.zeros((n_m, n_a), dtype=bool)
    match_periodo = np.zeros((n_m, n_a), dtype=bool)

    for i, m in mentoras.iterrows():
        m_inst = _normalizar_chave_comparacao(str(m.get("instituicao", "")))
        m_curso = _normalizar_chave_comparacao(str(m.get("curso", "")))
        m_per = float(m.get("ano_ingresso", 2026.1))
        for j, a in mentoradas.iterrows():
            a_inst = _normalizar_chave_comparacao(str(a.get("instituicao", "")))
            a_curso = _normalizar_chave_comparacao(str(a.get("curso", "")))
            a_per = float(a.get("ano_ingresso", 2026.1))

            afinidade[i, j] = calcular_afinidade_par(m, a, pesos=pesos_afinidade)
            match_inst[i, j] = m_inst == a_inst
            match_curso[i, j] = m_curso == a_curso
            match_periodo[i, j] = m_per < a_per

    return {
        "afinidade": afinidade,
        "inst": match_inst,
        "curso": match_curso,
        "periodo": match_periodo,
    }


def _mascara_candidatas_estruturais(
    matrizes: Dict[str, np.ndarray],
    j: int,
    regra_inst: bool,
    regra_curso: bool,
    regra_periodo: bool,
) -> np.ndarray:
    """Retorna índices de mentoras compatíveis com as regras ativadas para uma aluna."""
    n_m = matrizes["afinidade"].shape[0]
    mask = np.ones(n_m, dtype=bool)
    if regra_inst:
        mask &= matrizes["inst"][:, j]
    if regra_curso:
        mask &= matrizes["curso"][:, j]
    if regra_periodo:
        mask &= matrizes["periodo"][:, j]
    return np.where(mask)[0]


def gerar_rede_aleatoria(
    mentoras: pd.DataFrame,
    mentoradas: pd.DataFrame,
    matrizes: Dict[str, np.ndarray],
    regra_inst: bool = True,
    regra_curso: bool = True,
    regra_periodo: bool = True,
    max_par: int = 2,
    forcar_todas_alunas: bool = True,
    seed: int = 42,
) -> np.ndarray:
    """Gera uma alocação aleatória com capacidade e regras configuráveis."""
    rng = np.random.default_rng(seed)
    n_m, n_a = len(mentoras), len(mentoradas)
    if n_m == 0 or n_a == 0:
        return np.array([], dtype=int)

    max_par = max(1, int(max_par))
    alocacao = np.full(n_a, -1, dtype=int)
    carga_mentoras = np.zeros(n_m, dtype=int)

    indices = np.arange(n_a)
    rng.shuffle(indices)

    for j in indices:
        candidatas = _mascara_candidatas_estruturais(matrizes, j, regra_inst, regra_curso, regra_periodo)
        candidatas_livres = candidatas[carga_mentoras[candidatas] < max_par]

        if len(candidatas_livres) > 0:
            min_carga = carga_mentoras[candidatas_livres].min()
            pool = candidatas_livres[carga_mentoras[candidatas_livres] == min_carga]
            escolhida = int(rng.choice(pool))
            alocacao[j] = escolhida
            carga_mentoras[escolhida] += 1
        elif forcar_todas_alunas:
            if len(candidatas) > 0:
                escolhida = int(rng.choice(candidatas))
            else:
                livres_geral = np.where(carga_mentoras < max_par)[0]
                if len(livres_geral) > 0:
                    min_c = carga_mentoras[livres_geral].min()
                    pool_g = livres_geral[carga_mentoras[livres_geral] == min_c]
                    escolhida = int(rng.choice(pool_g))
                else:
                    escolhida = int(rng.choice(np.arange(n_m)))
            alocacao[j] = escolhida
            carga_mentoras[escolhida] += 1
        else:
            alocacao[j] = -1

    return alocacao


def avaliar_individuo(
    cromossomo: np.ndarray,
    matrizes: Dict[str, np.ndarray],
    regra_inst: bool = True,
    regra_curso: bool = True,
    regra_periodo: bool = True,
    max_par: int = 2,
    permitir_sem_par: bool = False,
    penalidades: Tuple[float, float, float] = (2.5, 1.5, 0.8),
) -> Tuple[float, float, float]:
    """Avalia um pareamento por afinidade média, violações e fitness escalar."""
    n_a = len(cromossomo)
    n_m = matrizes["afinidade"].shape[0]
    if n_a == 0 or n_m == 0:
        return 0.0, 0.0, 0.0

    pen_periodo, pen_inst, pen_curso_carga = penalidades
    max_par = max(1, int(max_par))
    validos_idx = np.where((cromossomo >= 0) & (cromossomo < n_m))[0]
    nao_atendidas = int(n_a - len(validos_idx))

    if len(validos_idx) > 0:
        m_idx = cromossomo[validos_idx]
        afinidades = matrizes["afinidade"][m_idx, validos_idx]
        v_inst = (~matrizes["inst"][m_idx, validos_idx]) if regra_inst else np.zeros(len(validos_idx), dtype=bool)
        v_curso = (~matrizes["curso"][m_idx, validos_idx]) if regra_curso else np.zeros(len(validos_idx), dtype=bool)
        v_periodo = (~matrizes["periodo"][m_idx, validos_idx]) if regra_periodo else np.zeros(len(validos_idx), dtype=bool)
        cargas = np.bincount(m_idx, minlength=n_m)
        excesso_carga = int(np.maximum(0, cargas - max_par).sum())
    else:
        afinidades = np.array([0.0], dtype=float)
        v_inst = np.array([], dtype=bool)
        v_curso = np.array([], dtype=bool)
        v_periodo = np.array([], dtype=bool)
        excesso_carga = 0

    penalidade_nao_atend = (0.0 if permitir_sem_par else 3.0) * nao_atendidas
    obj1_afinidade = float(np.mean(afinidades))
    obj2_penalidade = float(
        pen_periodo * float(v_periodo.sum())
        + pen_inst * float(v_inst.sum())
        + pen_curso_carga * float(v_curso.sum())
        + pen_curso_carga * float(excesso_carga)
        + penalidade_nao_atend
    ) / max(1, n_a)

    fitness_escalar = obj1_afinidade - obj2_penalidade
    return fitness_escalar, obj1_afinidade, obj2_penalidade


def _gerar_individuo_inicial(
    rng: np.random.Generator,
    mentoras: pd.DataFrame,
    mentoradas: pd.DataFrame,
    matrizes: Dict[str, np.ndarray],
    regra_inst: bool,
    regra_curso: bool,
    regra_periodo: bool,
    max_par: int,
    forcar_todas_alunas: bool,
    seed: int,
) -> np.ndarray:
    """Cria um pareamento inicial respeitando as regras e a capacidade disponível."""
    n_m, n_a = len(mentoras), len(mentoradas)
    if not forcar_todas_alunas:
        ind = np.full(n_a, -1, dtype=int)
        cargas = np.zeros(n_m, dtype=int)
        ordem = rng.permutation(n_a)
        for j in ordem:
            cands = _mascara_candidatas_estruturais(matrizes, int(j), regra_inst, regra_curso, regra_periodo)
            livres = cands[cargas[cands] < max_par]
            if len(livres) > 0:
                scores_c = matrizes["afinidade"][livres, int(j)] + rng.uniform(0.0, 0.05, size=len(livres))
                escolhida = int(livres[int(np.argmax(scores_c))])
                ind[int(j)] = escolhida
                cargas[escolhida] += 1
        return ind

    return gerar_rede_aleatoria(
        mentoras,
        mentoradas,
        matrizes,
        regra_inst=regra_inst,
        regra_curso=regra_curso,
        regra_periodo=regra_periodo,
        max_par=max_par,
        forcar_todas_alunas=True,
        seed=seed,
    )


def _mutar_cromossomo(
    rng: np.random.Generator,
    cromossomo: np.ndarray,
    n_m: int,
    taxa_mutacao: float,
    forcar_todas_alunas: bool,
) -> np.ndarray:
    """Aplica mutações aleatórias às alocações de um cromossomo."""
    filho = cromossomo.copy()
    mut_mask = rng.random(len(filho)) < taxa_mutacao
    if not np.any(mut_mask):
        return filho
    if forcar_todas_alunas:
        filho[mut_mask] = rng.integers(0, n_m, size=int(mut_mask.sum()))
    else:
        for idx in np.where(mut_mask)[0]:
            filho[idx] = int(rng.integers(-1, n_m))
    return filho


def executar_ag_mono_ou_elitista(
    mentoras: pd.DataFrame,
    mentoradas: pd.DataFrame,
    matrizes: Dict[str, np.ndarray],
    regra_inst: bool = True,
    regra_curso: bool = True,
    regra_periodo: bool = True,
    pop_size: int = 70,
    geracoes: int = 85,
    taxa_mutacao: float = 0.08,
    elitismo: bool = True,
    max_par: int = 2,
    forcar_todas_alunas: bool = True,
    penalidades: Tuple[float, float, float] = (2.5, 1.5, 0.8),
    seed: int = 42,
) -> np.ndarray:
    """Executa algoritmo genético mono-objetivo, opcionalmente preservando elites."""
    n_m, n_a = len(mentoras), len(mentoradas)
    if n_m == 0 or n_a == 0:
        return np.array([], dtype=int)

    rng = np.random.default_rng(seed)
    max_par = max(1, int(max_par))

    populacao = np.array(
        [
            _gerar_individuo_inicial(
                rng,
                mentoras,
                mentoradas,
                matrizes,
                regra_inst,
                regra_curso,
                regra_periodo,
                max_par=max_par,
                forcar_todas_alunas=forcar_todas_alunas,
                seed=seed + i,
            )
            for i in range(pop_size)
        ]
    )

    n_elite = max(2, int(pop_size * 0.1)) if elitismo else 0

    for _ in range(geracoes):
        scores = np.array(
            [
                avaliar_individuo(
                    ind,
                    matrizes,
                    regra_inst,
                    regra_curso,
                    regra_periodo,
                    max_par=max_par,
                    permitir_sem_par=not forcar_todas_alunas,
                    penalidades=penalidades,
                )[0]
                for ind in populacao
            ]
        )
        nova_pop: List[np.ndarray] = []
        if elitismo:
            elite_idx = np.argsort(scores)[-n_elite:]
            nova_pop.extend(populacao[elite_idx])

        while len(nova_pop) < pop_size:
            t1 = rng.choice(pop_size, size=min(3, pop_size), replace=False)
            p1 = populacao[t1[int(np.argmax(scores[t1]))]]
            t2 = rng.choice(pop_size, size=min(3, pop_size), replace=False)
            p2 = populacao[t2[int(np.argmax(scores[t2]))]]

            filho = np.where(rng.random(n_a) < 0.5, p1, p2)
            filho = _mutar_cromossomo(rng, filho, n_m, taxa_mutacao, forcar_todas_alunas)
            nova_pop.append(filho)

        populacao = np.array(nova_pop[:pop_size])

    scores_finais = np.array(
        [
            avaliar_individuo(
                ind,
                matrizes,
                regra_inst,
                regra_curso,
                regra_periodo,
                max_par=max_par,
                permitir_sem_par=not forcar_todas_alunas,
                penalidades=penalidades,
            )[0]
            for ind in populacao
        ]
    )
    return populacao[int(np.argmax(scores_finais))]


def executar_nsga2(
    mentoras: pd.DataFrame,
    mentoradas: pd.DataFrame,
    matrizes: Dict[str, np.ndarray],
    regra_inst: bool = True,
    regra_curso: bool = True,
    regra_periodo: bool = True,
    pop_size: int = 70,
    geracoes: int = 75,
    taxa_mutacao: float = 0.08,
    max_par: int = 2,
    forcar_todas_alunas: bool = True,
    penalidades: Tuple[float, float, float] = (2.5, 1.5, 0.8),
    seed: int = 42,
) -> np.ndarray:
    """Executa NSGA-II para equilibrar afinidade e penalidades do pareamento."""
    n_m, n_a = len(mentoras), len(mentoradas)
    if n_m == 0 or n_a == 0:
        return np.array([], dtype=int)

    rng = np.random.default_rng(seed)
    max_par = max(1, int(max_par))

    populacao = np.array(
        [
            _gerar_individuo_inicial(
                rng,
                mentoras,
                mentoradas,
                matrizes,
                regra_inst,
                regra_curso,
                regra_periodo,
                max_par=max_par,
                forcar_todas_alunas=forcar_todas_alunas,
                seed=seed + i,
            )
            for i in range(pop_size)
        ]
    )

    def non_dominated_sort(objs: np.ndarray) -> List[List[int]]:
        """Agrupa os indivíduos em frentes de dominância de Pareto."""
        N = len(objs)
        S: List[List[int]] = [[] for _ in range(N)]
        n = np.zeros(N, dtype=int)
        fronts: List[List[int]] = [[]]
        for p in range(N):
            for q in range(N):
                if (objs[p, 0] <= objs[q, 0] and objs[p, 1] <= objs[q, 1]) and (
                    objs[p, 0] < objs[q, 0] or objs[p, 1] < objs[q, 1]
                ):
                    S[p].append(q)
                elif (objs[q, 0] <= objs[p, 0] and objs[q, 1] <= objs[p, 1]) and (
                    objs[q, 0] < objs[p, 0] or objs[q, 1] < objs[p, 1]
                ):
                    n[p] += 1
            if n[p] == 0:
                fronts[0].append(p)
        i = 0
        while i < len(fronts) and fronts[i]:
            next_front: List[int] = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        return [f for f in fronts if len(f) > 0]

    for _ in range(geracoes):
        filhos: List[np.ndarray] = []
        for _ in range(pop_size):
            p1, p2 = populacao[rng.choice(pop_size, min(2, pop_size), replace=False)]
            filho = np.where(rng.random(n_a) < 0.5, p1, p2)
            filho = _mutar_cromossomo(rng, filho, n_m, taxa_mutacao, forcar_todas_alunas)
            filhos.append(filho)
        combinada = np.vstack([populacao, np.array(filhos)])

        avaliacoes = [
            avaliar_individuo(
                ind,
                matrizes,
                regra_inst,
                regra_curso,
                regra_periodo,
                max_par=max_par,
                permitir_sem_par=not forcar_todas_alunas,
                penalidades=penalidades,
            )
            for ind in combinada
        ]
        objs = np.array([[-a[1], a[2]] for a in avaliacoes])

        fronts = non_dominated_sort(objs)
        nova_pop: List[np.ndarray] = []
        for front in fronts:
            if len(nova_pop) + len(front) <= pop_size:
                nova_pop.extend(combinada[front])
            else:
                f_objs = objs[front]
                dist = np.zeros(len(front))
                for m_idx in range(2):
                    order = np.argsort(f_objs[:, m_idx])
                    dist[order[0]] = dist[order[-1]] = np.inf
                    span = f_objs[order[-1], m_idx] - f_objs[order[0], m_idx] + 1e-9
                    for k in range(1, len(front) - 1):
                        dist[order[k]] += (f_objs[order[k + 1], m_idx] - f_objs[order[k - 1], m_idx]) / span
                selecionados = np.argsort(dist)[-(pop_size - len(nova_pop)) :]
                nova_pop.extend(combinada[np.array(front)[selecionados]])
                break
        populacao = np.array(nova_pop[:pop_size])

    avaliacoes_finais = [
        avaliar_individuo(
            ind,
            matrizes,
            regra_inst,
            regra_curso,
            regra_periodo,
            max_par=max_par,
            permitir_sem_par=not forcar_todas_alunas,
            penalidades=penalidades,
        )
        for ind in populacao
    ]
    objs_finais = np.array([[-a[1], a[2]] for a in avaliacoes_finais])
    pareto_idx = non_dominated_sort(objs_finais)[0]

    f_pareto = objs_finais[pareto_idx]
    norm = (f_pareto - f_pareto.min(axis=0)) / (np.ptp(f_pareto, axis=0) + 1e-9)
    melhor_idx_pareto = pareto_idx[int(np.argmin(np.linalg.norm(norm, axis=1)))]

    return populacao[melhor_idx_pareto]


def _valido_sob_regras(
    i: int,
    j: int,
    matrizes: Dict[str, np.ndarray],
    regra_inst: bool,
    regra_curso: bool,
    regra_periodo: bool,
) -> bool:
    """Verifica se um par atende a todas as regras estruturais ativadas."""
    if regra_inst and not bool(matrizes["inst"][i, j]):
        return False
    if regra_curso and not bool(matrizes["curso"][i, j]):
        return False
    if regra_periodo and not bool(matrizes["periodo"][i, j]):
        return False
    return True


def executar_pareamento_peso_maximo_bipartido(
    mentoras: pd.DataFrame,
    mentoradas: pd.DataFrame,
    matrizes: Dict[str, np.ndarray],
    regra_inst: bool = True,
    regra_curso: bool = True,
    regra_periodo: bool = True,
    max_par: int = 2,
    forcar_todas_alunas: bool = True,
) -> np.ndarray:
    """Aloca mentoradas por matching de peso máximo em um grafo bipartido."""
    n_m, n_a = len(mentoras), len(mentoradas)
    if n_m == 0 or n_a == 0:
        return np.array([], dtype=int)

    max_par = max(1, int(max_par))
    G_bip = nx.Graph()

    for i in range(n_m):
        for c in range(max_par):
            u_slot = f"U_{i}_S{c}"
            G_bip.add_node(u_slot, bipartite=0, m_idx=i)

    for j in range(n_a):
        v_node = f"V_{j}"
        G_bip.add_node(v_node, bipartite=1, a_idx=j)

    for i in range(n_m):
        for j in range(n_a):
            valido = _valido_sob_regras(i, j, matrizes, regra_inst, regra_curso, regra_periodo)
            if not valido and not forcar_todas_alunas:
                continue
            af = float(matrizes["afinidade"][i, j])
            pen = 0.0 if valido else 0.50
            peso = max(0.001, (af + 0.10) - pen)
            for c in range(max_par):
                G_bip.add_edge(f"U_{i}_S{c}", f"V_{j}", weight=peso)

    alocacao = np.full(n_a, -1, dtype=int)
    if G_bip.number_of_edges() == 0:
        return alocacao

    matching = nx.max_weight_matching(
        G_bip,
        maxcardinality=True,
        weight="weight",
    )
    for no_a, no_b in matching:
        if G_bip.nodes[no_a]["bipartite"] == 1:
            no_a, no_b = no_b, no_a
        indice_mentorada = int(G_bip.nodes[no_b]["a_idx"])
        alocacao[indice_mentorada] = int(G_bip.nodes[no_a]["m_idx"])

    return alocacao


def executar_gulosa_afinidade_balanceada(
    mentoras: pd.DataFrame,
    mentoradas: pd.DataFrame,
    matrizes: Dict[str, np.ndarray],
    regra_inst: bool = True,
    regra_curso: bool = True,
    regra_periodo: bool = True,
    max_par: int = 2,
    forcar_todas_alunas: bool = True,
) -> np.ndarray:
    """Aloca pares em ordem de afinidade, priorizando regras e carga disponível."""
    n_m, n_a = len(mentoras), len(mentoradas)
    if n_m == 0 or n_a == 0:
        return np.array([], dtype=int)

    max_par = max(1, int(max_par))
    alocacao = np.full(n_a, -1, dtype=int)
    cargas = np.zeros(n_m, dtype=int)

    candidatos_ordenados: List[Tuple[float, int, int]] = []
    for j in range(n_a):
        for i in range(n_m):
            valido = _valido_sob_regras(i, j, matrizes, regra_inst, regra_curso, regra_periodo)
            if not valido and not forcar_todas_alunas:
                continue
            af = float(matrizes["afinidade"][i, j])
            score_ord = af + (0.25 if valido else -0.50)
            candidatos_ordenados.append((score_ord, i, j))

    candidatos_ordenados.sort(key=lambda item: (-item[0], item[1], item[2]))

    for _, i, j in candidatos_ordenados:
        if alocacao[j] != -1:
            continue
        if cargas[i] < max_par:
            alocacao[j] = i
            cargas[i] += 1

    return alocacao


def aplicar_flag_todas_alunas(
    aloc: np.ndarray,
    mentoras: pd.DataFrame,
    mentoradas: pd.DataFrame,
    matrizes: Dict[str, np.ndarray],
    regra_inst: bool,
    regra_curso: bool,
    regra_periodo: bool,
    max_par: int,
    forcar_todas_alunas: bool,
) -> np.ndarray:
    """Valida e completa a alocação conforme a opção de atender todas as alunas."""
    aloc_arr = np.asarray(aloc, dtype=int).copy()
    n_m = len(mentoras)
    n_a = len(mentoradas)
    max_par = max(1, int(max_par))
    if n_m == 0 or n_a == 0:
        return np.array([], dtype=int)

    if len(aloc_arr) != n_a:
        aloc_arr = np.full(n_a, -1, dtype=int)

    cargas = np.zeros(n_m, dtype=int)

    if not forcar_todas_alunas:
        for j in range(n_a):
            i = int(aloc_arr[j])
            if 0 <= i < n_m and cargas[i] < max_par and _valido_sob_regras(i, j, matrizes, regra_inst, regra_curso, regra_periodo):
                cargas[i] += 1
            else:
                aloc_arr[j] = -1

        for j in range(n_a):
            if aloc_arr[j] < 0:
                cands = _mascara_candidatas_estruturais(matrizes, j, regra_inst, regra_curso, regra_periodo)
                livres = cands[cargas[cands] < max_par]
                if len(livres) > 0:
                    escolhida = int(livres[int(np.argmax(matrizes["afinidade"][livres, j]))])
                    aloc_arr[j] = escolhida
                    cargas[escolhida] += 1
        return aloc_arr

    for j in range(n_a):
        i = int(aloc_arr[j])
        if 0 <= i < n_m and cargas[i] < max_par:
            cargas[i] += 1
        else:
            aloc_arr[j] = -1

    for j in range(n_a):
        if aloc_arr[j] < 0:
            cands = _mascara_candidatas_estruturais(matrizes, j, regra_inst, regra_curso, regra_periodo)
            livres_cands = cands[cargas[cands] < max_par]
            if len(livres_cands) > 0:
                escolhida = int(livres_cands[int(np.argmax(matrizes["afinidade"][livres_cands, j]))])
            elif len(cands) > 0:
                escolhida = int(cands[int(np.argmin(cargas[cands]))])
            else:
                livres_geral = np.where(cargas < max_par)[0]
                if len(livres_geral) > 0:
                    scores_g = matrizes["afinidade"][livres_geral, j] - 0.15 * cargas[livres_geral]
                    escolhida = int(livres_geral[int(np.argmax(scores_g))])
                else:
                    scores_all = matrizes["afinidade"][:, j] - 0.20 * cargas
                    escolhida = int(np.argmax(scores_all))
            aloc_arr[j] = escolhida
            cargas[escolhida] += 1

    return aloc_arr


def _classificar_status_aresta(
    ok_inst: bool,
    ok_curso: bool,
    ok_per: bool,
    excedeu_limite: bool,
) -> str:
    """Classifica o atendimento às regras e ao limite de carga de uma aresta."""
    regras_ok = ok_inst and ok_curso and ok_per
    if regras_ok and not excedeu_limite:
        return "Exato (Inst + Curso + Período)"
    if regras_ok and excedeu_limite:
        return "Exato c/ Excesso de Carga"
    if ok_inst or ok_per:
        return "Parcial c/ Flexibilização" if excedeu_limite else "Parcial"
    return "Fallback c/ Excesso de Carga" if excedeu_limite else "Fallback"


def construir_grafo_e_metricas(
    mentoras: pd.DataFrame,
    mentoradas: pd.DataFrame,
    alocacao: np.ndarray,
    matrizes: Dict[str, np.ndarray],
    max_par: int = 2,
) -> Tuple[nx.Graph, Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Constrói a rede de pares e calcula métricas gerais e por instituição."""
    G = nx.Graph()
    max_par = max(1, int(max_par))

    for _, m in mentoras.iterrows():
        G.add_node(
            m["id_anonimo"],
            papel=m["papel"],
            grupo_bipartido="U",
            instituicao=m["instituicao"],
            curso=m["curso"],
            ano_ingresso=m["ano_ingresso"],
            areas=str(m.get("areas_interesse", "Não informado")),
        )

    for _, a in mentoradas.iterrows():
        G.add_node(
            a["id_anonimo"],
            papel=a["papel"],
            grupo_bipartido="V",
            instituicao=a["instituicao"],
            curso=a["curso"],
            ano_ingresso=a["ano_ingresso"],
            areas=str(a.get("areas_interesse", "Não informado")),
        )

    uso_atual_mentora: Dict[int, int] = {}
    for j, i in enumerate(alocacao):
        idx_m = int(i)
        if idx_m < 0 or idx_m >= len(mentoras) or j < 0 or j >= len(mentoradas):
            continue
        m_id = mentoras.loc[idx_m, "id_anonimo"]
        a_id = mentoradas.loc[j, "id_anonimo"]

        uso_atual_mentora[idx_m] = uso_atual_mentora.get(idx_m, 0) + 1
        excedeu_limite = uso_atual_mentora[idx_m] > max_par

        ok_inst = bool(matrizes["inst"][idx_m, j])
        ok_curso = bool(matrizes["curso"][idx_m, j])
        ok_per = bool(matrizes["periodo"][idx_m, j])
        af = float(matrizes["afinidade"][idx_m, j])

        status = _classificar_status_aresta(ok_inst, ok_curso, ok_per, excedeu_limite)

        G.add_edge(
            m_id,
            a_id,
            afinidade=af,
            match_inst=ok_inst,
            match_curso=ok_curso,
            match_periodo=ok_per,
            excedeu_limite=excedeu_limite,
            status=status,
        )

    nos_ativos = [n for n, deg in G.degree() if deg > 0]
    G_conectado = G.subgraph(nos_ativos).copy()

    def calcular_bloco_metricas(subG: nx.Graph, escopo_label: str) -> Dict[str, Any]:
        """Calcula métricas bipartidas para um subgrafo e seu rótulo de escopo."""
        u_sub = [n for n, d in subG.nodes(data=True) if d["grupo_bipartido"] == "U"]
        v_sub = [n for n, d in subG.nodes(data=True) if d["grupo_bipartido"] == "V"]
        graus_sub = dict(subG.degree())
        n_edges = subG.number_of_edges()

        grau_medio_geral = float(np.mean(list(graus_sub.values()))) if graus_sub else 0.0
        grau_medio_u = float(np.mean([graus_sub[n] for n in u_sub])) if u_sub else 0.0
        denom_bip = len(u_sub) * len(v_sub)
        densidade_bip = float(n_edges / denom_bip) if denom_bip > 0 else 0.0

        if u_sub:
            hub_id = max(u_sub, key=lambda n: (graus_sub[n], n))
            hub_info = f"{hub_id} (k={graus_sub[hub_id]})"
        else:
            hub_info = "-"

        n_comp = nx.number_connected_components(subG) if subG.number_of_nodes() > 0 else 0
        afin_med = float(np.mean([d["afinidade"] for _, _, d in subG.edges(data=True)])) if n_edges else 0.0
        ader_per = 100.0 * float(np.mean([d["match_periodo"] for _, _, d in subG.edges(data=True)])) if n_edges else 0.0
        ader_inst = 100.0 * float(np.mean([d["match_inst"] for _, _, d in subG.edges(data=True)])) if n_edges else 0.0
        ader_curso = 100.0 * float(np.mean([d["match_curso"] for _, _, d in subG.edges(data=True)])) if n_edges else 0.0

        dp_grau_u = float(np.std([graus_sub[n] for n in u_sub])) if len(u_sub) > 1 else 0.0
        n_v_total_alvo = max(1, len(mentoradas))
        cobertura_v_pct = 100.0 * (len(v_sub) / n_v_total_alvo)

        return {
            "Escopo": escopo_label,
            "Nós Conectados": f"{subG.number_of_nodes()} ({len(u_sub)}U + {len(v_sub)}V)",
            "Cobertura V (%)": round(cobertura_v_pct, 1),
            "Pares (|E|)": n_edges,
            "Grau Médio (<k>)": round(grau_medio_geral, 2),
            "Grau Médio U": round(grau_medio_u, 2),
            "DP Grau U": round(dp_grau_u, 2),
            "Densidade Bipartida": round(densidade_bip, 4),
            "Hub": hub_info,
            "Componentes": n_comp,
            "Afinidade Média": round(afin_med, 3),
            "Período (%)": round(ader_per, 1),
            "Instituição (%)": round(ader_inst, 1),
            "Curso (%)": round(ader_curso, 1),
        }

    metricas_geral = calcular_bloco_metricas(G_conectado, "Todas as Instituições (Geral)")

    instituicoes_no_grafo = sorted({d["instituicao"] for _, d in G_conectado.nodes(data=True)})
    metricas_por_inst: Dict[str, Dict[str, Any]] = {}
    for inst in instituicoes_no_grafo:
        nos_inst = [n for n, d in G_conectado.nodes(data=True) if d["instituicao"] == inst]
        sub_inst = G_conectado.subgraph(nos_inst).copy()
        metricas_por_inst[inst] = calcular_bloco_metricas(sub_inst, inst)

    return G_conectado, metricas_geral, metricas_por_inst


def gerar_resumo_textual_complementar(
    G: nx.Graph,
    total_mentoradas_filtro: int,
    relatorio_ingestao: Optional[RelatorioIngestao] = None,
) -> html.Div:
    """Monta um resumo visual da cobertura, afinidade, regras e avisos da ingestão."""
    n_edges = G.number_of_edges()
    v_conectadas = sum(1 for _, d in G.nodes(data=True) if d["grupo_bipartido"] == "V")

    elementos: List[Any] = []

    if relatorio_ingestao and relatorio_ingestao.avisos:
        elementos.append(
            dbc.Alert(
                [
                    html.Strong("Auditoria da base: "),
                    " | ".join(relatorio_ingestao.avisos[:3]),
                ],
                color="info",
                style={"marginBottom": "8px", "fontSize": "0.80rem", "padding": "6px 10px"},
            )
        )

    if n_edges == 0:
        elementos.append(
            dbc.Alert(
                f"Nenhum par formado com os filtros atuais. Alunas conectadas: 0 de {total_mentoradas_filtro}.",
                color="warning",
                style={"marginBottom": "0", "fontSize": "0.88rem"},
            )
        )
        return html.Div(elementos)

    graus_m = [G.degree(n) for n, d in G.nodes(data=True) if d["grupo_bipartido"] == "U"]
    if graus_m:
        u_vals, u_cnt = np.unique(graus_m, return_counts=True)
        dist_grau_str = ", ".join([f"grau {k}: {c} mentora(s)" for k, c in zip(u_vals, u_cnt)])
    else:
        dist_grau_str = "sem mentoras conectadas"

    scores = [float(d["afinidade"]) for _, _, d in G.edges(data=True)]
    alta_cnt = sum(1 for s in scores if s >= 0.60)
    media_cnt = sum(1 for s in scores if 0.35 <= s < 0.60)
    baixa_cnt = sum(1 for s in scores if s < 0.35)
    afin_med = float(np.mean(scores))
    afin_min = float(np.min(scores))
    afin_max = float(np.max(scores))

    ader_per = 100.0 * float(np.mean([d["match_periodo"] for _, _, d in G.edges(data=True)]))
    ader_inst = 100.0 * float(np.mean([d["match_inst"] for _, _, d in G.edges(data=True)]))
    ader_curso = 100.0 * float(np.mean([d["match_curso"] for _, _, d in G.edges(data=True)]))

    pares_parciais_ou_fallback = sum(
        1
        for _, _, d in G.edges(data=True)
        if not (d["match_periodo"] and d["match_inst"] and d["match_curso"]) or d.get("excedeu_limite", False)
    )

    elementos.extend(
        [
            html.P(
                [
                    html.Strong("Cobertura das Alunas (V): "),
                    f"{v_conectadas} de {total_mentoradas_filtro} aluna(s) conectada(s) em {n_edges} par(es).",
                ],
                style={"fontSize": "0.86rem", "marginBottom": "6px"},
            ),
            html.P(
                [
                    html.Strong("Distribuição do Grau das Mentoras (U): "),
                    f"{dist_grau_str}.",
                ],
                style={"fontSize": "0.86rem", "marginBottom": "6px"},
            ),
            html.P(
                [
                    html.Strong("Escore de Afinidade por Par: "),
                    f"média {afin_med:.4f} (mín. {afin_min:.4f}, máx. {afin_max:.4f}) — "
                    f"alta (≥ 0,60): {alta_cnt} par(es), média (0,35 a 0,59): {media_cnt} par(es), baixa (< 0,35): {baixa_cnt} par(es).",
                ],
                style={"fontSize": "0.86rem", "marginBottom": "6px"},
            ),
            html.P(
                [
                    html.Strong("Aderência às Regras: "),
                    f"Período (>1 sem.): {ader_per:.1f}%, Instituição: {ader_inst:.1f}%, Curso: {ader_curso:.1f}%.",
                ],
                style={"fontSize": "0.86rem", "marginBottom": "6px"},
            ),
            html.P(
                [
                    html.Strong("Pares com Exceção / Flexibilização: "),
                    f"{pares_parciais_ou_fallback} de {n_edges} par(es) apresentam alguma regra não atendida ou limite excedido.",
                ],
                style={"fontSize": "0.84rem", "marginBottom": "0", "color": "#495057"},
            ),
        ]
    )

    return html.Div(elementos)


def _assinatura_grafo_layout(G: nx.Graph, tipo_grafico: str) -> str:
    """Gera uma chave determinística para reutilizar o layout de um grafo."""
    nos = sorted((n, round(float(d.get("ano_ingresso", 0.0)), 2), str(d.get("instituicao", ""))) for n, d in G.nodes(data=True))
    arestas = sorted((u, v, round(float(d.get("afinidade", 0.0)), 4)) for u, v, d in G.edges(data=True))
    payload = f"{tipo_grafico}|{nos}|{arestas}".encode("utf-8")
    return hashlib.md5(payload).hexdigest()


_CACHE_POSICOES_GRAFO: Dict[str, Dict[str, Tuple[float, float]]] = {}


def _obter_posicoes_grafo(G: nx.Graph, tipo_grafico: str) -> Dict[str, Tuple[float, float]]:
    """Obtém do cache ou calcula as coordenadas dos nós para o tipo de visualização."""
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
    """Cria a figura Plotly dos pares, com detalhes de nós e conexões no hover."""
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

    def montar_trace_papel(lista_nos: List[str], nome_legenda: str, cor_no: str, simbolo_no: str, tamanho_base: int) -> go.Scatter:
        """Cria a camada de nós de um papel com tamanho e informações interativas."""
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


# ==============================================================================
# 2. LÓGICA DE COMUNIDADES (REDE E COMUNIDADES)
# ==============================================================================

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
    """Gera uma base sintética reprodutível para a análise de comunidades."""
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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(caminho_csv):
        caminho_csv = os.path.join(base_dir, caminho_csv)
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
    """Calcula o peso de uma conexão segundo o critério selecionado."""
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
    """Monta o grafo direcionado de participantes e conexões ponderadas."""
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
    """Retorna o subgrafo correspondente aos filtros de instituição e turno."""
    df_filtrado = df.copy()
    if instituicao != "GERAL":
        df_filtrado = df_filtrado[df_filtrado["instituicao"] == instituicao]
    if turno != "TODOS":
        df_filtrado = df_filtrado[df_filtrado["turno"] == turno]
    ids_selecionados = df_filtrado["id_aluna"].tolist()
    return grafo.subgraph(ids_selecionados).copy()


def _converter_para_nao_direcionado_ponderado(grafo: nx.DiGraph) -> nx.Graph:
    """Converte o grafo em não direcionado, somando pesos de arestas recíprocas."""
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
    """Seleciona partição por modularidade ponderada com cortes por distância inversa."""
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
    """Detecta comunidades com o algoritmo escolhido e calcula sua modularidade."""
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
    """Resume tamanho, turno predominante e nó mais conectado de cada comunidade."""
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


CAMINHO_CSV_COMUNIDADES = r"data/base_conexoes_switab_ficticia.csv"
df_comunidades, origem_comunidades = carregar_dados_comunidades(CAMINHO_CSV_COMUNIDADES)
grafo_comunidades_base = construir_grafo_completo(df_comunidades, criterio_peso="composto")


@lru_cache(maxsize=64)
def _calcular_tabela_comparativa_cached(
    criterio_peso: str, filtro_instituicao: str
) -> Tuple[Dict[str, object], ...]:
    """Calcula e armazena em cache métricas de algoritmos por instituição."""
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
            comunidades, modularidade, nome_alg = detectar_comunidades(subgrafo, algoritmo=alg_cod)
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
    """Retorna como lista as métricas comparativas armazenadas no cache."""
    return list(_calcular_tabela_comparativa_cached(criterio_peso, filtro_instituicao))


def criar_figura_grafo(
    grafo: nx.DiGraph,
    titulo: str,
    comunidades: List[Set[str]],
    nome_algoritmo: str,
    exibir_setas: bool = True,
) -> go.Figure:
    """Cria a visualização Plotly da rede com cores por comunidade."""
    if grafo.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text=titulo, font=dict(size=18, color="#1F2937")),
            template="plotly_white",
            height=540,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

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
        height=580,
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
    )
    return fig


def criar_card_metrica(titulo: str, valor: object, subtitulo: str = "") -> html.Div:
    """Cria um cartão visual para exibir uma métrica e sua descrição opcional."""
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


# ==============================================================================
# 3. CACHE DE PAREAMENTO E LAYOUT COM DOIS PAINÉIS (PAREAMENTO E COMUNIDADES)
# ==============================================================================

df_global, relatorio_global = carregar_dados_base_com_relatorio()
instituicoes_individuais = sorted(df_global["instituicao"].dropna().astype(str).str.strip().unique().tolist())
opcoes_filtro_geral_inst = [{"label": "Todas as Instituições", "value": "TODAS"}] + [
    {"label": inst, "value": inst} for inst in instituicoes_individuais
]

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
    """Gera uma chave determinística baseada nos parâmetros da simulação."""
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
    """Executa ou recupera do cache o pareamento e suas métricas."""
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


painel_pareamento = dbc.Container(
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
                                    html.Label("Instituição (Flags)", style={"fontWeight": "600", "fontSize": "0.85rem"}),
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
                                        value="nsga2",
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
                                    html.Label("Curso", style={"fontWeight": "600", "fontSize": "0.85rem"}),
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
                                            value=2,
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
                                                        value=0.7,
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
                                                        value=0.15,
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
                                                        value=0.15,
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
                                                        value=2.5,
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
                                                        value=1.5,
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
                                                        value=0.8,
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

painel_comunidades = dbc.Container(
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
                        html.Div(id="conteudo-painel-ativo"),
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
    """Alterna a visibilidade dos painéis de pareamento e comunidades."""
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
    """Inverte a visibilidade do painel de personalização."""
    return not is_open


@app.callback(
    Output("collapse-caracteristicas", "is_open"),
    Input("btn-toggle-caracteristicas", "n_clicks"),
    State("collapse-caracteristicas", "is_open"),
    prevent_initial_call=True,
)
def toggle_caracteristicas(n_clicks: Optional[int], is_open: bool) -> bool:
    """Inverte a visibilidade da tabela de características da rede."""
    return not is_open


@app.callback(
    Output("collapse-explicacao", "is_open"),
    Input("btn-toggle-explicacao", "n_clicks"),
    State("collapse-explicacao", "is_open"),
    prevent_initial_call=True,
)
def toggle_explicacao(n_clicks: Optional[int], is_open: bool) -> bool:
    """Inverte a visibilidade da explicação dos algoritmos."""
    return not is_open


@app.callback(
    Output("collapse-explicacao-metricas", "is_open"),
    Input("btn-toggle-explicacao-metricas", "n_clicks"),
    State("collapse-explicacao-metricas", "is_open"),
    prevent_initial_call=True,
)
def toggle_explicacao_metricas(n_clicks: Optional[int], is_open: bool) -> bool:
    """Inverte a visibilidade da explicação das métricas."""
    return not is_open


@app.callback(
    Output("collapse-sugestoes", "is_open"),
    Input("btn-toggle-sugestoes", "n_clicks"),
    State("collapse-sugestoes", "is_open"),
    prevent_initial_call=True,
)
def toggle_sugestoes(n_clicks: Optional[int], is_open: bool) -> bool:
    """Inverte a visibilidade das sugestões de filtros."""
    return not is_open


@app.callback(
    Output("collapse-pares", "is_open"),
    Input("btn-toggle-pares", "n_clicks"),
    State("collapse-pares", "is_open"),
    prevent_initial_call=True,
)
def toggle_pares(n_clicks: Optional[int], is_open: bool) -> bool:
    """Inverte a visibilidade da tabela de pares."""
    return not is_open


@app.callback(
    Output("download-csv", "data"),
    Input("btn-exportar-csv", "n_clicks"),
    State("tabela-pares", "data"),
    State("dropdown-modelo", "value"),
    prevent_initial_call=True,
)
def exportar_csv(n_clicks: Optional[int], dados_tabela: Optional[List[Dict[str, Any]]], modelo: Optional[str]) -> Any:
    """Prepara o download CSV dos pares exibidos, identificando o algoritmo no nome."""
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
    """Atualiza as opções de curso conforme as instituições selecionadas."""
    inst_lista = inst_flags_sel if isinstance(inst_flags_sel, list) else ([inst_flags_sel] if inst_flags_sel else ["TODAS"])
    inst_clean = [str(x).strip() for x in inst_lista if str(x).strip()]
    individuais = [x for x in inst_clean if x != "TODAS"]

    df_sub = df_global.copy()
    if individuais:
        alvo_inst = {_normalizar_chave_comparacao(x) for x in individuais}
        df_sub = df_sub[df_sub["instituicao"].apply(lambda v: _normalizar_chave_comparacao(v) in alvo_inst)]

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
    """Obtém métricas da rede completa e filtra as linhas pelas instituições marcadas."""
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
        chaves_alvo.add(_normalizar_chave_comparacao(f_inst))

    if not chaves_alvo:
        return todas_linhas

    filtradas = [
        linha for linha in todas_linhas
        if _normalizar_chave_comparacao(str(linha.get("Escopo", ""))) in chaves_alvo
    ]
    return filtradas if filtradas else todas_linhas


@app.callback(
    Output("container-tabela-metricas", "style"),
    Input("flag-exibir-metricas-acima-csv", "value"),
)
def toggle_exibir_metricas_acima_csv(flag_val: Any) -> Dict[str, str]:
    """Define a visibilidade da tabela de métricas conforme a opção selecionada."""
    ativos = flag_val if isinstance(flag_val, list) else ([flag_val] if flag_val else [])
    if "exibir_acima_csv" in ativos:
        return {"display": "block"}
    return {"display": "none"}


def _parse_float_seguro(val: Any, padrao: float) -> float:
    """Converte um valor para float não negativo ou retorna o padrão informado."""
    if val is None or val == "":
        return padrao
    try:
        num = float(val)
        return num if np.isfinite(num) and num >= 0.0 else padrao
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
) -> Tuple[go.Figure, html.Div, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Atualiza gráfico, resumo e tabelas do painel conforme filtros e parâmetros."""
    inst_lista = inst_flags_sel if isinstance(inst_flags_sel, list) else ([inst_flags_sel] if inst_flags_sel else ["TODAS"])
    cursos_lista = cursos_flags_sel if isinstance(cursos_flags_sel, list) else ([cursos_flags_sel] if cursos_flags_sel else ["TODOS_CURSOS"])
    tipo_graf = tipo_grafico or "similaridade_espacial"
    alg = modelo or "nsga2"
    limite_par = max(1, int(max_par or 2))

    pesos_afinidade = (
        _parse_float_seguro(peso_areas_val, 0.70),
        _parse_float_seguro(peso_inst_val, 0.15),
        _parse_float_seguro(peso_curso_val, 0.15),
    )
    penalidades = (
        _parse_float_seguro(pen_periodo_val, 2.5),
        _parse_float_seguro(pen_inst_val, 1.5),
        _parse_float_seguro(pen_curso_carga_val, 0.8),
    )

    regra_periodo = "periodo" in (regras_sel or [])
    regra_inst = "inst" in (regras_sel or [])
    regra_curso = "curso" in (regras_sel or [])
    forcar_todas_alunas = bool(flag_todas_sel) and (
        "todas_alunas" in (flag_todas_sel if isinstance(flag_todas_sel, list) else [flag_todas_sel])
    )

    caract_linhas = construir_caracteristicas_gerais_rede(df_global, inst_lista, cursos_lista)

    G_con, met_geral, met_por_inst, total_mentoradas = rodar_simulacao_cacheada(
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
    """Atualiza a rede de comunidades, seus cartões de métricas e a tabela-resumo."""
    grafo_ponderado = construir_grafo_completo(df_comunidades, criterio_peso=criterio_peso)
    subgrafo = extrair_subgrafo_filtrado(
        grafo_ponderado, df_comunidades, instituicao_selecionada, turno=turno_selecionado
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
    """Atualiza a tabela comparativa para a instituição e o peso selecionados."""
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
    """Gera o arquivo CSV com os resultados comparativos das comunidades."""
    dados = gerar_tabela_comparativa(
        criterio_peso=criterio_peso, filtro_instituicao=filtro_tabela
    )
    df_export = pd.DataFrame(dados)
    buffer = io.StringIO()
    df_export.to_csv(buffer, index=False, encoding="utf-8-sig")
    return dict(content=buffer.getvalue(), filename="comparativo_algoritmos_switab.csv")


if __name__ == "__main__":
    modo_debug = os.environ.get("SWITAB_DEBUG", "false").strip().lower() in {"1", "true", "yes"}
    porta = int(os.environ.get("PORT", "8050"))
    app.run(debug=modo_debug, host="127.0.0.1", port=porta)
