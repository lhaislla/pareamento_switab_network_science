"""Carregamento e validacao de configuracoes a partir de variaveis de ambiente (.env)."""

from dataclasses import dataclass
import os
from dotenv import load_dotenv


def _parse_bool(valor: str | None, padrao: bool = False) -> bool:
    """Converte string de ambiente em booleano de forma segura."""
    if valor is None:
        return padrao
    return valor.strip().lower() in {"1", "true", "yes", "on", "sim"}


def _parse_float(valor: str | None, padrao: float) -> float:
    """Converte string de ambiente em float com fallback seguro."""
    if valor is None or not valor.strip():
        return padrao
    try:
        num = float(valor.strip().replace(",", "."))
        return num if num >= 0.0 else padrao
    except ValueError:
        return padrao


def _parse_int(valor: str | None, padrao: int) -> int:
    """Converte string de ambiente em int com fallback seguro."""
    if valor is None or not valor.strip():
        return padrao
    try:
        return int(valor.strip())
    except ValueError:
        return padrao


@dataclass(frozen=True)
class ConfiguracaoApp:
    """Estrutura imutavel de configuracoes da aplicacao."""

    debug: bool
    host: str
    porta: int
    seed: int
    caminho_dados_pareamento: str
    caminho_dados_comunidades: str
    peso_areas: float
    peso_instituicao: float
    peso_curso: float
    penalidade_periodo: float
    penalidade_instituicao: float
    penalidade_curso_carga: float
    limite_conexoes_padrao: int


def obter_configuracao(raiz_projeto: str) -> ConfiguracaoApp:
    """Carrega variaveis do arquivo .env (se presente) e retorna ConfiguracaoApp."""
    env_path = os.path.join(raiz_projeto, ".env")
    if os.path.isfile(env_path):
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        load_dotenv(override=False)

    rel_par = os.getenv("PAREAMENTO_DATA_PATH", "data/SWITAB.csv").strip()
    rel_com = os.getenv("COMUNIDADES_DATA_PATH", "data/base_conexoes_switab_ficticia.csv").strip()

    caminho_par = rel_par if os.path.isabs(rel_par) else os.path.join(raiz_projeto, rel_par)
    caminho_com = rel_com if os.path.isabs(rel_com) else os.path.join(raiz_projeto, rel_com)

    return ConfiguracaoApp(
        debug=_parse_bool(os.getenv("SWITAB_DEBUG"), padrao=False),
        host=os.getenv("SWITAB_HOST", "127.0.0.1").strip() or "127.0.0.1",
        porta=max(1, _parse_int(os.getenv("PORT"), padrao=8050)),
        seed=_parse_int(os.getenv("SWITAB_SEED"), padrao=42),
        caminho_dados_pareamento=caminho_par,
        caminho_dados_comunidades=caminho_com,
        peso_areas=_parse_float(os.getenv("PESO_AREAS_JACCARD"), padrao=0.70),
        peso_instituicao=_parse_float(os.getenv("PESO_INSTITUICAO"), padrao=0.15),
        peso_curso=_parse_float(os.getenv("PESO_CURSO"), padrao=0.15),
        penalidade_periodo=_parse_float(os.getenv("PENALIDADE_PERIODO"), padrao=2.5),
        penalidade_instituicao=_parse_float(os.getenv("PENALIDADE_INSTITUICAO"), padrao=1.5),
        penalidade_curso_carga=_parse_float(os.getenv("PENALIDADE_CURSO_CARGA"), padrao=0.8),
        limite_conexoes_padrao=max(1, _parse_int(os.getenv("LIMITE_CONEXOES_PADRAO"), padrao=2)),
    )
