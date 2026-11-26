import hashlib
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import dash_bootstrap_components as dbc
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html


# ==============================================================================
# 1. CONFIGURAÇÃO, NORMALIZAÇÃO E VALIDAÇÃO DE DADOS
# ==============================================================================

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
    sem_acento = "".join(
        ch for ch in unicodedata.normalize("NFKD", str(col).strip().lower())
        if not unicodedata.combining(ch)
    )
    return re.sub(r"[^a-z0-9]+", "_", sem_acento).strip("_")


def _normalizar_texto(valor: Any, padrao: str = "Não Informado") -> str:
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return padrao
    texto = re.sub(r"\s+", " ", str(valor).strip())
    if not texto or texto.lower() in {"nan", "none", "null", "n/a", "na"}:
        return padrao
    return texto


def _normalizar_chave_comparacao(valor: str) -> str:
    sem_acento = "".join(
        ch for ch in unicodedata.normalize("NFKD", str(valor).strip().lower())
        if not unicodedata.combining(ch)
    )
    return re.sub(r"\s+", " ", sem_acento)


def _parse_periodo_ingresso(valor: Any, padrao: float = 2026.1) -> float:
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
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    candidatos: List[str] = []
    if caminho_arquivo:
        candidatos.append(caminho_arquivo)

    candidatos.extend(
        [
            os.path.join(base_dir, "data", "SWITAB.csv"),
            os.path.join(os.getcwd(), "data", "SWITAB.csv"),
            os.path.join(base_dir, "dados_mentoria_anonimizados.csv"),
            "dados_mentoria_anonimizados.csv",
        ]
    )

    for path in candidatos:
        if path and os.path.isfile(path):
            try:
                df_raw = _ler_arquivo_tabular(path)
                df_limpo, rel = _validar_e_limpar_dataframe(df_raw, origem=f"Arquivo: {path}")
                if len(df_limpo) > 0:
                    return df_limpo, rel
                rel.avisos.append("Arquivo encontrado, mas sem linhas válidas; usando dataset sintético.")
                df_sint = gerar_dataset_sintetico(seed=42)
                df_limpo_s, _ = _validar_e_limpar_dataframe(df_sint, origem="Sintético (fallback arquivo vazio)")
                rel.total_linhas_validas = len(df_limpo_s)
                return df_limpo_s, rel
            except Exception as exc:
                df_sint = gerar_dataset_sintetico(seed=42)
                df_limpo_s, rel_s = _validar_e_limpar_dataframe(
                    df_sint,
                    origem=f"Sintético (erro ao ler '{path}': {exc})",
                )
                rel_s.avisos.append(f"Falha ao ler '{path}': {exc}. Executando com base sintética.")
                return df_limpo_s, rel_s

    df_sint = gerar_dataset_sintetico(seed=42)
    df_limpo_s, rel_s = _validar_e_limpar_dataframe(df_sint, origem="Sintético (SWITAB.csv não encontrado)")
    rel_s.avisos.append("Arquivo data/SWITAB.csv não localizado; base sintética carregada para simulação.")
    return df_limpo_s, rel_s


def carregar_dados_base(caminho_arquivo: Optional[str] = None) -> pd.DataFrame:
    df_limpo, _ = carregar_dados_base_com_relatorio(caminho_arquivo=caminho_arquivo)
    return df_limpo


def construir_caracteristicas_gerais_rede(
    df: pd.DataFrame,
    instituicoes_selecionadas: Optional[List[str]] = None,
    cursos_selecionados: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
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
    w_jaccard, w_inst, w_curso = pesos
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


def construir_matriz_compatibilidade(mentoras: pd.DataFrame, mentoradas: pd.DataFrame) -> Dict[str, np.ndarray]:
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

            afinidade[i, j] = calcular_afinidade_par(m, a)
            match_inst[i, j] = m_inst == a_inst
            match_curso[i, j] = m_curso == a_curso
            match_periodo[i, j] = m_per < a_per

    return {
        "afinidade": afinidade,
        "inst": match_inst,
        "curso": match_curso,
        "periodo": match_periodo,
    }


# ==============================================================================
# 2. ALGORITMOS DE PAREAMENTO E TRATAMENTO EXPLÍCITO DE REGRAS / CAPACIDADE
# ==============================================================================

def _mascara_candidatas_estruturais(
    matrizes: Dict[str, np.ndarray],
    j: int,
    regra_inst: bool,
    regra_curso: bool,
    regra_periodo: bool,
) -> np.ndarray:
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
) -> Tuple[float, float, float]:
    n_a = len(cromossomo)
    n_m = matrizes["afinidade"].shape[0]
    if n_a == 0 or n_m == 0:
        return 0.0, 0.0, 0.0

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
        2.5 * float(v_periodo.sum())
        + 1.5 * float(v_inst.sum())
        + 0.8 * float(v_curso.sum())
        + 2.0 * float(excesso_carga)
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
    seed: int = 42,
) -> np.ndarray:
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
    seed: int = 42,
) -> np.ndarray:
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
        )
        for ind in populacao
    ]
    objs_finais = np.array([[-a[1], a[2]] for a in avaliacoes_finais])
    pareto_idx = non_dominated_sort(objs_finais)[0]

    f_pareto = objs_finais[pareto_idx]
    norm = (f_pareto - f_pareto.min(axis=0)) / (np.ptp(f_pareto, axis=0) + 1e-9)
    melhor_idx_pareto = pareto_idx[int(np.argmin(np.linalg.norm(norm, axis=1)))]

    return populacao[melhor_idx_pareto]


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
    """
    Modela o pareamento como um grafo bipartido ponderado e resolve via
    algoritmo de emparelhamento de peso máximo (nx.bipartite.maximum_matching),
    replicando nós de mentoras por capacidade (max_par) para encontrar o
    ótimo combinatório exato sobre os pesos de afinidade e aderência às regras.
    """
    n_m, n_a = len(mentoras), len(mentoradas)
    if n_m == 0 or n_a == 0:
        return np.array([], dtype=int)

    max_par = max(1, int(max_par))
    G_bip = nx.Graph()
    nos_u = []
    nos_v = []

    for i in range(n_m):
        for c in range(max_par):
            u_slot = f"U_{i}_S{c}"
            nos_u.append(u_slot)
            G_bip.add_node(u_slot, bipartite=0, m_idx=i)

    for j in range(n_a):
        v_node = f"V_{j}"
        nos_v.append(v_node)
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

    matching = nx.bipartite.maximum_matching(G_bip, top_nodes=nos_u)
    for j in range(n_a):
        v_node = f"V_{j}"
        parceiro = matching.get(v_node)
        if parceiro and parceiro in G_bip.nodes:
            alocacao[j] = int(G_bip.nodes[parceiro]["m_idx"])

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
    """
    Heurística construtiva determinística de Ciência das Redes:
    ordena pares (mentora, aluna) por maior afinidade e menor carga atual
    da mentora, priorizando aderência às regras estruturais ativas.
    """
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


def _valido_sob_regras(
    i: int,
    j: int,
    matrizes: Dict[str, np.ndarray],
    regra_inst: bool,
    regra_curso: bool,
    regra_periodo: bool,
) -> bool:
    if regra_inst and not bool(matrizes["inst"][i, j]):
        return False
    if regra_curso and not bool(matrizes["curso"][i, j]):
        return False
    if regra_periodo and not bool(matrizes["periodo"][i, j]):
        return False
    return True


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
    """
    Aplica a regra de cobertura das alunas:
    - forcar_todas_alunas=True: garante par para todas as alunas (V). Quando todas as
      mentoras atingem max_par ou não há par sob todas as regras, escolhe a melhor
      alternativa disponível registrando a condição via status da aresta.
    - forcar_todas_alunas=False: mantém apenas alocações estritamente válidas pelas
      regras ativas e pelo limite max_par, e tenta reaproveitar candidatas válidas
      livres para alunas elegíveis ainda sem par.
    """
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
                livres_geral = np.where(carga_mentoras < max_par)[0] if False else np.where(cargas < max_par)[0]
                if len(livres_geral) > 0:
                    scores_g = matrizes["afinidade"][livres_geral, j] - 0.15 * cargas[livres_geral]
                    escolhida = int(livres_geral[int(np.argmax(scores_g))])
                else:
                    scores_all = matrizes["afinidade"][:, j] - 0.20 * cargas
                    escolhida = int(np.argmax(scores_all))
            aloc_arr[j] = escolhida
            cargas[escolhida] += 1

    return aloc_arr


# ==============================================================================
# 3. GRAFO CONECTADO, MÉTRICAS, AUDITORIA DE EXCEÇÕES E RESUMO TEXTUAL
# ==============================================================================

def _classificar_status_aresta(
    ok_inst: bool,
    ok_curso: bool,
    ok_per: bool,
    excedeu_limite: bool,
) -> str:
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

    carga_por_mentora: Dict[int, int] = {}
    for j, i in enumerate(alocacao):
        if 0 <= int(i) < len(mentoras) and 0 <= j < len(mentoradas):
            idx_m = int(i)
            carga_por_mentora[idx_m] = carga_por_mentora.get(idx_m, 0) + 1

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


def gerar_resumo_textual_complementar(
    G: nx.Graph,
    total_mentoradas_filtro: int,
    relatorio_ingestao: Optional[RelatorioIngestao] = None,
) -> html.Div:
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


# ==============================================================================
# 4. GRÁFICO PRINCIPAL SEM SOBREPOSIÇÃO DE TÍTULO
# ==============================================================================

def _assinatura_grafo_layout(G: nx.Graph, tipo_grafico: str) -> str:
    nos = sorted((n, round(float(d.get("ano_ingresso", 0.0)), 2), str(d.get("instituicao", ""))) for n, d in G.nodes(data=True))
    arestas = sorted((u, v, round(float(d.get("afinidade", 0.0)), 4)) for u, v, d in G.edges(data=True))
    payload = f"{tipo_grafico}|{nos}|{arestas}".encode("utf-8")
    return hashlib.md5(payload).hexdigest()


_CACHE_POSICOES_GRAFO: Dict[str, Dict[str, Tuple[float, float]]] = {}


def _obter_posicoes_grafo(G: nx.Graph, tipo_grafico: str) -> Dict[str, Tuple[float, float]]:
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
# 5. CACHE DE EXECUÇÃO E APLICAÇÃO PLOTLY DASH
# ==============================================================================

df_global, relatorio_global = carregar_dados_base_com_relatorio()
instituicoes_individuais = sorted(df_global["instituicao"].dropna().astype(str).str.strip().unique().tolist())
instituicoes_disponiveis = ["Todas as Instituições"] + instituicoes_individuais
opcoes_filtro_geral_inst = [{"label": "Todas as Instituições", "value": "TODAS"}] + [
    {"label": inst, "value": inst} for inst in instituicoes_individuais
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
) -> str:
    chave = (
        f"{tuple(sorted(inst_lista))}|{tuple(sorted(cursos_lista))}|{modelo}|"
        f"{tuple(sorted(papeis_sel or []))}|{regra_inst}|{regra_curso}|{regra_periodo}|"
        f"{forcar_todas_alunas}|{max_par}"
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
) -> Tuple[nx.Graph, Dict[str, Any], Dict[str, Dict[str, Any]], int]:
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
    )
    if chave in _CACHE_SIMULACAO:
        G_c, m_ger_c, m_inst_c = _CACHE_SIMULACAO[chave]
        return G_c, m_ger_c, m_inst_c, total_v

    matrizes = construir_matriz_compatibilidade(mentoras, mentoradas)

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


app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="SWITAB - Dashboard de Pareamento",
)

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3(
                            "SWITAB - Dashboard de Pareamento",
                            style={"fontWeight": "700", "color": "#4a148c", "marginTop": "18px", "marginBottom": "4px"},
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
                                            {"label": "Emparelhamento Bipartido de Peso Máximo", "value": "bipartite_max_matching"},
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
                                            "quantidade total de participantes ativos na rede formada, separando mentoras (U) e participantes recebidas (V).",
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
                                            html.Strong("3. Emparelhamento Bipartido de Peso Máximo (NetworkX): "),
                                            "representa as capacidades das mentoras como slots em um grafo bipartido ponderado e aplica correspondência de peso máximo. ",
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
    style={"backgroundColor": "#f4f6f9", "minHeight": "100vh"},
)


@app.callback(
    Output("collapse-caracteristicas", "is_open"),
    Input("btn-toggle-caracteristicas", "n_clicks"),
    State("collapse-caracteristicas", "is_open"),
    prevent_initial_call=True,
)
def toggle_caracteristicas(n_clicks: Optional[int], is_open: bool) -> bool:
    return not is_open


@app.callback(
    Output("collapse-explicacao", "is_open"),
    Input("btn-toggle-explicacao", "n_clicks"),
    State("collapse-explicacao", "is_open"),
    prevent_initial_call=True,
)
def toggle_explicacao(n_clicks: Optional[int], is_open: bool) -> bool:
    return not is_open


@app.callback(
    Output("collapse-explicacao-metricas", "is_open"),
    Input("btn-toggle-explicacao-metricas", "n_clicks"),
    State("collapse-explicacao-metricas", "is_open"),
    prevent_initial_call=True,
)
def toggle_explicacao_metricas(n_clicks: Optional[int], is_open: bool) -> bool:
    return not is_open


@app.callback(
    Output("collapse-sugestoes", "is_open"),
    Input("btn-toggle-sugestoes", "n_clicks"),
    State("collapse-sugestoes", "is_open"),
    prevent_initial_call=True,
)
def toggle_sugestoes(n_clicks: Optional[int], is_open: bool) -> bool:
    return not is_open


@app.callback(
    Output("collapse-pares", "is_open"),
    Input("btn-toggle-pares", "n_clicks"),
    State("collapse-pares", "is_open"),
    prevent_initial_call=True,
)
def toggle_pares(n_clicks: Optional[int], is_open: bool) -> bool:
    return not is_open


@app.callback(
    Output("download-csv", "data"),
    Input("btn-exportar-csv", "n_clicks"),
    State("tabela-pares", "data"),
    State("dropdown-modelo", "value"),
    prevent_initial_call=True,
)
def exportar_csv(n_clicks: Optional[int], dados_tabela: Optional[List[Dict[str, Any]]], modelo: Optional[str]) -> Any:
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
    flags_metricas_inst: Any = None,
) -> List[Dict[str, Any]]:
    """
    Calcula as métricas de forma totalmente isolada do filtro geral de instituição
    e curso do topo, obedecendo exclusivamente ao filtro próprio da seção de Métricas
    (dropdown e flags de instituições) para exibir o resumo acima da Tabela de Pares / CSV.
    """
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
    ativos = flag_val if isinstance(flag_val, list) else ([flag_val] if flag_val else [])
    if "exibir_acima_csv" in ativos:
        return {"display": "block"}
    return {"display": "none"}


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
    ],
)
def atualizar_dashboard(
    inst_flags_sel: Any,
    cursos_flags_sel: Any,
    tipo_grafico: Optional[str],
    modelo: Optional[str],
    papeis_sel: Optional[List[str]],
    regras_sel: Optional[List[str]],
    flag_todas_sel: Any,
    max_par: Optional[int],
    flags_metricas_inst: Any = None,
) -> Tuple[go.Figure, html.Div, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    inst_lista = inst_flags_sel if isinstance(inst_flags_sel, list) else ([inst_flags_sel] if inst_flags_sel else ["TODAS"])
    cursos_lista = cursos_flags_sel if isinstance(cursos_flags_sel, list) else ([cursos_flags_sel] if cursos_flags_sel else ["TODOS_CURSOS"])
    tipo_graf = tipo_grafico or "similaridade_espacial"
    alg = modelo or "nsga2"
    limite_par = max(1, int(max_par or 2))

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


if __name__ == "__main__":
    modo_debug = os.environ.get("SWITAB_DEBUG", "false").strip().lower() in {"1", "true", "yes"}
    porta = int(os.environ.get("PORT", "8050"))
    app.run(debug=modo_debug, host="127.0.0.1", port=porta)
