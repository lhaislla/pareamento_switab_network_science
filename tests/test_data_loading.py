import tempfile
import unittest
from pathlib import Path

import pandas as pd

from core.comunidades_engine import carregar_dados_comunidades
from core.pareamento_engine import carregar_dados_base_com_relatorio


class DataLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.data_dir = Path(self.temp_dir.name)

    def test_pairing_loader_reads_requested_file(self) -> None:
        path = self.data_dir / "pareamento.csv"
        pd.DataFrame(
            {
                "id_anonimo": ["P001"],
                "papel": ["Mentora"],
                "instituicao": ["UPE"],
                "curso": ["Computação"],
                "ano_ingresso": [2022.1],
            }
        ).to_csv(path, index=False)

        data, report = carregar_dados_base_com_relatorio(str(path))

        self.assertEqual(len(data), 1)
        self.assertEqual(data.loc[0, "id_anonimo"], "P001")
        self.assertEqual(report.origem, f"Arquivo: {path}")

    def test_pairing_loader_rejects_missing_file(self) -> None:
        path = self.data_dir / "ausente.csv"

        with self.assertRaises(FileNotFoundError):
            carregar_dados_base_com_relatorio(str(path))

    def test_pairing_loader_rejects_unrecognized_schema(self) -> None:
        path = self.data_dir / "invalido.csv"
        pd.DataFrame({"coluna_desconhecida": ["valor"]}).to_csv(path, index=False)

        with self.assertRaisesRegex(ValueError, "colunas obrigatórias"):
            carregar_dados_base_com_relatorio(str(path))

    def test_communities_loader_reads_requested_file(self) -> None:
        path = self.data_dir / "comunidades.csv"
        pd.DataFrame(
            {
                "id_aluna": ["A001"],
                "instituicao": ["UPE"],
                "turno": ["Noite"],
                "colegas_conectadas_ids": [""],
            }
        ).to_csv(path, index=False)

        data, origin = carregar_dados_comunidades(str(path))

        self.assertEqual(len(data), 1)
        self.assertEqual(data.loc[0, "id_aluna"], "A001")
        self.assertEqual(origin, f"Arquivo: {path}")

    def test_communities_loader_rejects_missing_file(self) -> None:
        path = self.data_dir / "ausente.csv"

        with self.assertRaises(FileNotFoundError):
            carregar_dados_comunidades(str(path))

    def test_communities_loader_rejects_unrecognized_schema(self) -> None:
        path = self.data_dir / "invalido.csv"
        pd.DataFrame({"coluna_desconhecida": ["valor"]}).to_csv(path, index=False)

        with self.assertRaisesRegex(ValueError, "colunas obrigatórias"):
            carregar_dados_comunidades(str(path))


if __name__ == "__main__":
    unittest.main()
