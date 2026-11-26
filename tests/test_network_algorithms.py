import unittest

import networkx as nx
import numpy as np
import pandas as pd

from core.comunidades_engine import (
    calcular_metricas_rede,
    calcular_proporcao_conexoes_mesmo_turno,
    detectar_comunidades,
)
from core.pareamento_engine import executar_pareamento_peso_maximo_bipartido


class NetworkAlgorithmTests(unittest.TestCase):
    def test_matching_maximizes_weight_with_maximum_cardinality(self) -> None:
        mentoras = pd.DataFrame({"id_anonimo": ["M0", "M1"]})
        mentoradas = pd.DataFrame({"id_anonimo": ["A0", "A1"]})
        regras = {
            "inst": np.ones((2, 2), dtype=bool),
            "curso": np.ones((2, 2), dtype=bool),
            "periodo": np.ones((2, 2), dtype=bool),
        }
        cases = (
            (np.array([[1.0, 0.0], [0.0, 1.0]]), [0, 1]),
            (np.array([[0.0, 1.0], [1.0, 0.0]]), [1, 0]),
        )

        for afinidade, expected in cases:
            with self.subTest(afinidade=afinidade.tolist()):
                alocacao = executar_pareamento_peso_maximo_bipartido(
                    mentoras,
                    mentoradas,
                    {**regras, "afinidade": afinidade},
                    max_par=1,
                    forcar_todas_alunas=False,
                )

                self.assertEqual(alocacao.tolist(), expected)

    def test_betweenness_interprets_edge_weight_as_strength(self) -> None:
        grafo = nx.DiGraph()
        grafo.add_nodes_from(
            (node, {"nome": node, "turno": "Noite"})
            for node in ("A", "B", "C")
        )
        grafo.add_edge("A", "B", weight=10.0)
        grafo.add_edge("B", "C", weight=10.0)
        grafo.add_edge("A", "C", weight=0.1)

        metricas = calcular_metricas_rede(grafo, [{"A", "B", "C"}], 0.0)

        self.assertEqual(metricas["principal_ponte"], "B (B)")
        self.assertGreater(metricas["betweenness_max"], 0.0)

    def test_community_algorithms_return_partitions_on_weighted_graph(self) -> None:
        grafo = nx.DiGraph()
        grafo.add_nodes_from(("A", "B", "C", "D"))
        grafo.add_weighted_edges_from(
            [
                ("A", "B", 3.0),
                ("B", "A", 3.0),
                ("C", "D", 3.0),
                ("D", "C", 3.0),
                ("B", "C", 0.1),
            ],
            weight="weight",
        )

        for algoritmo in (
            "louvain",
            "label_propagation",
            "greedy_modularity",
            "girvan_newman",
        ):
            with self.subTest(algoritmo=algoritmo):
                comunidades, modularidade, _ = detectar_comunidades(
                    grafo,
                    algoritmo=algoritmo,
                    seed=7,
                )
                membros = [no for comunidade in comunidades for no in comunidade]

                self.assertEqual(set(membros), set(grafo.nodes()))
                self.assertEqual(len(membros), len(set(membros)))
                self.assertTrue(np.isfinite(modularidade))

    def test_same_turn_share_excludes_edges_with_unknown_turn(self) -> None:
        grafo = nx.DiGraph()
        grafo.add_node("A", turno="Noite")
        grafo.add_node("B", turno="Noite")
        grafo.add_node("C", turno="")
        grafo.add_node("D", turno="Manhã")
        grafo.add_edges_from(
            [("A", "B"), ("B", "A"), ("A", "C"), ("A", "D"), ("D", "A")]
        )

        self.assertEqual(calcular_proporcao_conexoes_mesmo_turno(grafo), 50.0)

    def test_betweenness_rejects_nonpositive_strength(self) -> None:
        grafo = nx.DiGraph()
        grafo.add_edge("A", "B", weight=0.0)

        with self.assertRaisesRegex(ValueError, "deve ser positiva e finita"):
            calcular_metricas_rede(grafo, [{"A", "B"}], 0.0)


if __name__ == "__main__":
    unittest.main()
