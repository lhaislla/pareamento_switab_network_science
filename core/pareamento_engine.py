"""Funcoes de ingestao, validacao, afinidade, otimizacao e metricas para pareamento."""

from dataclasses import dataclass, field
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

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
    """Resumo de auditoria da leitura e saneamento da base de participantes."""

    origem: str
    total_linhas_brutas: int = 0
    total_linhas_validas: int = 0
    linhas_descartadas: int = 0
    colunas_mapeadas: Dict[str, str] = field(default_factory=dict)
    avisos: List[str] = field(default_factory=list)


def _normalizar_nome_coluna(col: str) -> str:
    """Remove acentos e padroniza nomes de colunas em snake_case."""
    sem_acento = "".join(
        ch for ch in unicodedata.normalize("NFKD", str(col).strip().lower())
        if not unicodedata.combining(ch)
    )
    return re.sub(r"[^a-z0-9]+", "_", sem_acento).strip("_")


def _normalizar_texto(valor: Any, padrao: str = "Não Informado") -> str:
    """Limpa espacos e trata valores vazios ou nulos retornando o padrao."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return padrao
    texto = re.sub(r"\s+", " ", str(valor).strip())
    if not texto or texto.lower() in {"nan", "none", "null", "n/a", "na"}:
        return padrao
    return texto


def normalizar_chave_comparacao(valor: str) -> str:
    """Normaliza string para comparacoes insensiveis a caixa e acentuacao."""
    sem_acento = "".join(
        ch for ch in unicodedata.normalize("NFKD", str(valor).strip().lower())
        if not unicodedata.combining(ch)
    )
    return re.sub(r"\s+", " ", sem_acento)


def _parse_periodo_ingresso(valor: Any, padrao: float = 2026.1) -> float:
    """Converte periodos academicos (ex.: 2024.1, 2024-2, 2025) para float AAAA.S."""
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
    """Extrai o conjunto normalizado de areas de interesse a partir de string separada por delimitadores."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return set()
    texto = str(valor).strip()
    if not texto or texto.lower() in {"nan", "none", "null", "não informado", "nao informado"}:
        return set()
    partes = re.split(r"[,;|/]+", texto)
    areas = set()
    for parte in partes:
        limpa = normalizar_chave_comparacao(parte)
        if limpa and limpa not in {"nan", "none", "null", "nao informado"}:
            areas.add(limpa)
    return areas


def _validar_e_limpar_dataframe(df_bruto: pd.DataFrame, origem: str) -> Tuple[pd.DataFrame, RelatorioIngestao]:
    """Padroniza colunas, trata ausentes/duplicatas e retorna DataFrame limpo com relatorio."""
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
            elif canonico in {"turno", "areas_interesse"}:
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
    """Gera base sintetica reproduzivel de mentoras, mentoradas e embaixadoras."""
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
    """Le arquivo CSV, TSV ou Excel com fallback de delimitador e codificacao."""
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
    """Carrega e valida a base de pareamento, sem substituí-la por dados sintéticos."""
    raiz_projeto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if caminho_arquivo is None:
        caminho_arquivo = os.path.join(raiz_projeto, "data", "SWITAB.csv")
    elif not os.path.isabs(caminho_arquivo):
        caminho_arquivo = os.path.join(raiz_projeto, caminho_arquivo)

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
    """Agrega caracteristicas demograficas, academicas e tematicas por instituicao e curso."""
    df_base, _ = _validar_e_limpar_dataframe(df, origem="filtro_caracteristicas")

    if instituicoes_selecionadas:
        inst_clean_sel = [str(x).strip() for x in instituicoes_selecionadas if str(x).strip()]
        individuais = [x for x in inst_clean_sel if x != "TODAS"]
        if individuais:
            alvo_inst = {normalizar_chave_comparacao(x) for x in individuais}
            df_base = df_base[df_base["instituicao"].apply(lambda v: normalizar_chave_comparacao(v) in alvo_inst)].copy()

    if cursos_selecionados:
        cursos_clean_sel = [str(x).strip() for x in cursos_selecionados if str(x).strip()]
        cursos_ind = [x for x in cursos_clean_sel if x != "TODOS_CURSOS"]
        if cursos_ind:
            alvo_curso = {normalizar_chave_comparacao(x) for x in cursos_ind}
            df_base = df_base[df_base["curso"].apply(lambda v: normalizar_chave_comparacao(v) in alvo_curso)].copy()

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
    """Separa os conjuntos de mentoras (U) e participantes recebidas (V) conforme filtros."""
    df_clean, _ = _validar_e_limpar_dataframe(df, origem="conjuntos_bipartidos")

    if instituicoes_selecionadas:
        inst_clean_sel = [str(x).strip() for x in instituicoes_selecionadas if str(x).strip()]
        individuais_marcadas = [x for x in inst_clean_sel if x != "TODAS"]
        if individuais_marcadas:
            alvo_inst = {normalizar_chave_comparacao(x) for x in individuais_marcadas}
            df_clean = df_clean[
                df_clean["instituicao"].apply(lambda v: normalizar_chave_comparacao(v) in alvo_inst)
            ].copy()

    if cursos_selecionados:
        cursos_clean_sel = [str(x).strip() for x in cursos_selecionados if str(x).strip()]
        cursos_individuais = [x for x in cursos_clean_sel if x != "TODOS_CURSOS"]
        if cursos_individuais:
            alvo_curso = {normalizar_chave_comparacao(x) for x in cursos_individuais}
            df_clean = df_clean[
                df_clean["curso"].apply(lambda v: normalizar_chave_comparacao(v) in alvo_curso)
            ].copy()

    if not papeis_selecionados:
        papeis_selecionados = ["Mentora", "Mentorada"]

    papeis_lower = {normalizar_chave_comparacao(p) for p in papeis_selecionados}
    df_filtrado_papel = df_clean[df_clean["papel"].apply(lambda p: normalizar_chave_comparacao(p) in papeis_lower)].copy()

    mentoras = df_filtrado_papel[df_filtrado_papel["papel"].apply(lambda p: normalizar_chave_comparacao(p) == "mentora")].copy()
    mentoradas = df_filtrado_papel[df_filtrado_papel["papel"].apply(lambda p: normalizar_chave_comparacao(p) != "mentora")].copy()

    return mentoras.reset_index(drop=True), mentoradas.reset_index(drop=True)


def calcular_afinidade_par(
    m_row: pd.Series,
    a_row: pd.Series,
    col_texto: str = "areas_interesse",
    pesos: Tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> float:
    """Calcula escore [0, 1] de afinidade combinando Jaccard de areas, instituicao e curso."""
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
        if normalizar_chave_comparacao(str(m_row.get("instituicao", "")))
        == normalizar_chave_comparacao(str(a_row.get("instituicao", "")))
        else 0.0
    )
    bonus_curso = (
        w_curso
        if normalizar_chave_comparacao(str(m_row.get("curso", "")))
        == normalizar_chave_comparacao(str(a_row.get("curso", "")))
        else 0.0
    )

    score = w_jaccard * jaccard + bonus_inst + bonus_curso
    return float(np.clip(round(score, 4), 0.0, 1.0))


def construir_matriz_compatibilidade(
    mentoras: pd.DataFrame,
    mentoradas: pd.DataFrame,
    pesos_afinidade: Tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> Dict[str, np.ndarray]:
    """Constroi matrizes de afinidade e aderencia estrutural (instituicao, curso, periodo)."""
    n_m, n_a = len(mentoras), len(mentoradas)
    afinidade = np.zeros((n_m, n_a), dtype=float)
    match_inst = np.zeros((n_m, n_a), dtype=bool)
    match_curso = np.zeros((n_m, n_a), dtype=bool)
    match_periodo = np.zeros((n_m, n_a), dtype=bool)

    for i, m in mentoras.iterrows():
        m_inst = normalizar_chave_comparacao(str(m.get("instituicao", "")))
        m_curso = normalizar_chave_comparacao(str(m.get("curso", "")))
        m_per = float(m.get("ano_ingresso", 2026.1))
        for j, a in mentoradas.iterrows():
            a_inst = normalizar_chave_comparacao(str(a.get("instituicao", "")))
            a_curso = normalizar_chave_comparacao(str(a.get("curso", "")))
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
    """Retorna indices de mentoras validas para a participante j sob as regras ativas."""
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
    """Gera alocacao inicial aleatoria respeitando capacidade e regras quando possivel."""
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
    """Avalia um vetor de alocacao retornando (fitness_escalar, afinidade_media, penalidade_media)."""
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
    """Cria individuo inicial para populacao genetica respeitando ou nao alocacao obrigatoria."""
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
    """Aplica mutacao aleatoria pontual ao cromossomo."""
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
    """Executa algoritmo genetico mono-objetivo com ou sem elitismo."""
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
    """Executa NSGA-II multiobjetivo (afinidade vs penalidade) e seleciona solucao de compromisso."""
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
    """Verifica se o par (i, j) atende todas as regras estruturais ativas."""
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
    """Resolve pareamento via emparelhamento bipartido ponderado com slots de capacidade."""
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
    """Executa alocacao gulosa deterministica priorizando afinidade, regras e capacidade."""
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
    """Ajusta alocacao final para garantir ou nao cobertura total das participantes em V."""
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
    """Rotula o par formado conforme aderencia as regras e capacidade da mentora."""
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
    """Constroi o grafo bipartido de pares ativos e calcula metricas gerais e por instituicao."""
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
