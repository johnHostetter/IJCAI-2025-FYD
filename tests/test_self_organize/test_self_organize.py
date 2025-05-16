"""
Test the self-organizing procedures of the soft computing library, such as self-organizing fuzzy
reinforcement learning.
"""

import os
import pathlib
import unittest
from typing import Dict, Union

import torch
from regime import Regime, Resource, Node
from fuzzy.relations.t_norm import Product
from fuzzy.logic.variables import LinguisticVariables
from fuzzy.logic.knowledge_base import KnowledgeBase

# the following algorithms are eligible for self-organizing neuro-fuzzy networks
from fuzzy_ml.utils import set_rng
from fuzzy_ml.datasets import LabeledDataset
from fuzzy_ml.fetchers import fetch_labeled_dataset
from fuzzy_ml.clustering.ecm import EvolvingClusteringMethod
from fuzzy_ml.clustering.empirical import find_empirical_fuzzy_sets
from fuzzy_ml.partitioning.clip import CategoricalLearningInducedPartitioning
from fuzzy_ml.rulemaking.wang_mendel import WangMendelMethod as WM

from soft_computing.utilities.reproducibility import load_configuration


AVAILABLE_DEVICE = torch.device("cpu")  # CUDA out of memory error for these tests


class TestSelfOrganize(unittest.TestCase):
    """
    The self-organizing process can be thought as a Knowledge Base (KB) constructing another KB.
    However, it passes the relevant components needed to call the expert design process when
    it has finished, to conclude the construction of the KB.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        directory = pathlib.Path(__file__).parent.resolve()
        file_path = os.path.join(directory, "small_data.pt")
        self.callables: Dict[str, Union[Node, callable]] = {
            "clip": CategoricalLearningInducedPartitioning(),
            "ecm": EvolvingClusteringMethod(),
            "kb": KnowledgeBase.create,
            "exemplars": fetch_labeled_dataset,
            "wm": WM(t_norm=Product),
        }  # either a function or a Node object are allowed
        self.data = torch.load(file_path).to(AVAILABLE_DEVICE)
        self.config = load_configuration()
        with self.config.unfreeze():
            self.config.device = "cpu"
            self.config.fuzzy_ml.clustering.ecm.distance_threshold = 1e-3

    def test_add_callables(self) -> None:
        """
        We can add an iterable collection of callable objects (e.g., functions, classes).

        Returns:
            None
        """
        regime = Regime(callables=set(self.callables.values()))
        assert len(regime.graph.vs) == len(self.callables)
        assert len(regime.graph.es) == 0  # we have not linked the functions yet
        # check that the required hyperparameters are as expected for the given callables
        # (i.e., the callables that have the 'hyperparameters' attribute)
        # value None means that the hyperparameter is not set
        assert regime.required_hyperparameters == {
            # follows the project structure
            "fuzzy_ml": {
                "partitioning": {
                    "clip": {
                        "CategoricalLearningInducedPartitioning": {
                            "epsilon": None,
                            "adjustment": None,
                        }
                    }
                },
                "clustering": {
                    "ecm": {"EvolvingClusteringMethod": {"distance_threshold": None}}
                },
            }
        }  # the configuration file passed to Regime MUST contain the above hyperparameters

    def test_load_hyperparameters(self) -> None:
        """
        Test that required hyperparameters are loaded correctly from the configuration provided.

        Returns:
            None
        """
        regime = Regime(callables=set(self.callables.values()))
        # create a config that satisfies the required hyperparameters
        config = load_configuration()
        defined_hyperparameters = regime.define_hyperparameters(configuration=config)
        assert defined_hyperparameters == {
            # follows the project structure
            "fuzzy_ml": {
                "partitioning": {
                    "clip": {
                        "CategoricalLearningInducedPartitioning": {
                            "epsilon": 0.6,
                            "adjustment": 0.2,
                        }
                    }
                },
                "clustering": {
                    "ecm": {"EvolvingClusteringMethod": {"distance_threshold": 0.7}}
                },
            }
        }  # the configuration file passed to Regime MUST contain the above hyperparameters

    def test_add_edges(self) -> None:
        """
        Test that adding several edges works as intended.

        Returns:
            None
        """
        regime = Regime(callables=set(self.callables.values()))
        edges = [
            (self.callables["ecm"], self.callables["wm"], 0),
            (self.callables["clip"], self.callables["wm"], 1),
            ("t_norm", self.callables["wm"], 2),
        ]
        # create a config that satisfies the required hyperparameters
        config = load_configuration()
        # set up the Regime object with the configuration
        regime.setup(configuration=config, resources=None, edges=None, clean_up=False)
        # link the processes together
        regime.define_flow(edges, clean_up=False)
        assert len(regime.graph.vs) == len(self.callables) + len(regime.resources)
        assert len(regime.graph.es) == len(edges)

        edges = [
            (self.callables["ecm"], self.callables["wm"], 0),
        ]
        self.assertRaises(ValueError, regime.define_flow, edges)

    def test_add_invalid_source_vertex(self) -> None:
        """
        Test that adding an edge with an invalid source vertex raises an error.

        Returns:
            None
        """
        callables = {
            self.callables["ecm"],
            find_empirical_fuzzy_sets,
            self.callables["wm"],
        }
        regime = Regime(callables=callables)
        edges = [
            (None, self.callables["wm"], 0),
            (self.callables["clip"], self.callables["wm"], 1),
            ("t_norm", self.callables["wm"], 2),
        ]
        # assert regime.link_functions(edges) throws a ValueError because of the None vertex
        self.assertRaises(ValueError, regime.define_flow, edges)
        # assert regime.link_functions(edges) throws a ValueError because it is missing the
        # CategoricalLearningInducedPartitioning in its 'functions' argument
        self.assertRaises(ValueError, regime.define_flow, edges[1:])

    def test_add_invalid_target_vertex(self) -> None:
        """
        Test that adding an edge with an invalid target vertex raises an error.

        Returns:
            None
        """
        callables = {
            self.callables["ecm"],
            self.callables["clip"],
        }
        regime = Regime(callables=callables)
        edges = [
            (self.callables["ecm"], None, 0),
            (self.callables["clip"], self.callables["wm"], 1),
            ("t_norm", self.callables["wm"], 2),
        ]
        # assert regime.link_functions(edges) throws a ValueError because of the None vertex
        self.assertRaises(ValueError, regime.define_flow, edges)
        # assert regime.link_functions(edges) throws a ValueError because it is missing the
        # WM in its 'functions' argument
        self.assertRaises(ValueError, regime.define_flow, edges[1:])

    def test_add_input_data(self) -> None:
        """
        Test adding a special vertex to store the input data. The special vertex's value is passed
        as an argument to functions that rely upon input data.

        Returns:
            None
        """
        regime = Regime(callables=set(self.callables.values()))
        edges = [
            (self.callables["ecm"], self.callables["wm"], 0),
            (self.callables["clip"], self.callables["wm"], 1),
            ("t_norm", self.callables["wm"], 2),
        ]
        regime.setup(
            configuration=self.config,
            resources={Resource(name="input", value=self.data)},
            edges=edges,
            clean_up=False,
        )
        assert len(regime.graph.vs) == len(self.callables) + len(
            regime.resources
        )  # (incl. data vertex)
        assert len(regime.graph.es) == len(edges)

        # test that we can decide to add more edges to the regime's workflow
        more_edges = [
            ("input", self.callables["ecm"], 0),
            ("input", self.callables["clip"], 0),
        ]
        regime.define_flow(more_edges)
        assert len(regime.graph.es) == len(edges) + len(more_edges)

    def test_get_kwargs(self) -> None:
        """
        Test that the keyword arguments are as expected in Regime.

        Returns:
            None
        """
        regime = Regime(callables=set(self.callables.values()))
        edges = self.callables["clip"].edges() + self.callables["ecm"].edges()
        regime.setup(
            configuration=self.config,
            resources={
                Resource(
                    name="train_dataset",
                    value=LabeledDataset(data=self.data, out_features=1),
                ),
                Resource(name="device", value=AVAILABLE_DEVICE),
            },
            edges=edges,
            clean_up=False,
        )
        assert len(regime.graph.vs) == len(self.callables) + len(
            regime.resources
        )  # (incl. data & config vertices)
        assert len(regime.graph.es) == len(edges)

        more_edges = self.callables["wm"].edges()
        regime.define_flow(more_edges)
        assert len(regime.graph.es) == len(edges) + len(
            more_edges
        )  # edges added to existing

        # find the vertex for this function & its predecessors
        target_vertex = regime.graph.vs.find(callable_eq=self.callables["wm"])

        expected_kwargs = {
            "exemplars": None,
            "linguistic_variables": None,
            "device": AVAILABLE_DEVICE,
        }
        self.assertEqual(expected_kwargs, regime.get_keyword_arguments(target_vertex))

    def test_unlinked_start(self) -> None:
        """
        Test that the self-organizing process raises an error if it is started without any
        vertices that are linked.

        Returns:
            None
        """
        regime = Regime(callables=set(self.callables.values()))
        self.assertRaises(ValueError, regime.start)

    def test_start(self) -> None:
        """
        Test a verbose definition of a self-organizing process (i.e., no shortcut method call).

        Returns:
            KnowledgeBase
        """
        set_rng(0)

        clip = CategoricalLearningInducedPartitioning()
        ecm = EvolvingClusteringMethod()
        wm = WM(t_norm=Product)
        regime = Regime(
            callables={
                clip,
                ecm,
                fetch_labeled_dataset,
                wm,
                KnowledgeBase.create,
            },
            resources={
                Resource(
                    name="train_dataset",
                    value=LabeledDataset(data=self.data, out_features=1),
                ),
                Resource(name="linguistic_variables", value=None),
                Resource(name="labeled_clusters", value=None),
                Resource(name="exemplars", value=None),
                Resource(name="rules", value=None),
                Resource(name="device", value=AVAILABLE_DEVICE),
            },
            verbose=True,
        )
        regime.setup(
            configuration=self.config,
            edges=clip.edges()
            + ecm.edges()
            + wm.edges()
            + [
                ("linguistic_variables", KnowledgeBase.create, 0),
                ("rules", KnowledgeBase.create, 1),
            ],
        )

        knowledge_base = regime.start()[
            f"{KnowledgeBase.create.__module__}.{KnowledgeBase.create.__name__}"
        ]

        # --- test info flowed properly from input data to CLIP ---
        actual_kwargs = regime.get_keyword_arguments(regime.get_vertex(clip))
        expected_kwargs = {
            "device": AVAILABLE_DEVICE,
            "train_dataset": LabeledDataset(data=self.data, out_features=1),
            "epsilon": 0.6,
            "adjustment": 0.2,
        }
        assert torch.allclose(
            actual_kwargs["train_dataset"].data, expected_kwargs["train_dataset"].data
        )

        # --- test info flowed properly from input data to ECM ---
        actual_kwargs = regime.get_keyword_arguments(regime.get_vertex(ecm))
        expected_kwargs = {
            "train_dataset": LabeledDataset(data=self.data, out_features=1),
            "distance_threshold": 0.7,
            "device": AVAILABLE_DEVICE,
        }
        assert torch.allclose(
            actual_kwargs["train_dataset"].data, expected_kwargs["train_dataset"].data
        )

        # --- test info flowed properly from ECM to fetch_fuzzy_set_centers ---
        ecm_output = regime.graph.vs.find(callable_eq=ecm)["output"]
        actual_kwargs = regime.get_keyword_arguments(
            regime.get_vertex(fetch_labeled_dataset)
        )
        expected_kwargs = {"labeled_clusters": ecm_output}
        assert torch.allclose(
            actual_kwargs["labeled_clusters"].get_centers(),
            expected_kwargs["labeled_clusters"].get_centers(),
        )
        assert torch.allclose(
            actual_kwargs["labeled_clusters"].get_widths(),
            expected_kwargs["labeled_clusters"].get_widths(),
        )

        # --- test info flowed properly from CLIP and fetch_fuzzy_set_centers to Wang-Mendel ---
        linguistic_variables: LinguisticVariables = regime.graph.vs.find(
            callable_eq=clip
        )["output"]
        dataset: LabeledDataset = regime.graph.vs.find(
            callable_eq=fetch_labeled_dataset
        )["output"]
        actual_kwargs = regime.get_keyword_arguments(regime.get_vertex(wm))
        expected_kwargs = {
            "exemplars": dataset,
            "linguistic_variables": linguistic_variables,
        }
        assert actual_kwargs["exemplars"] == expected_kwargs["exemplars"]
        assert (
            actual_kwargs["linguistic_variables"].inputs
            == expected_kwargs["linguistic_variables"].inputs
        )

        # --- test info flowed properly from CLIP and Wang-Mendel to expert_design ---
        linguistic_variables: LinguisticVariables = regime.graph.vs.find(
            callable_eq=clip
        )["output"]
        rules = regime.graph.vs.find(callable_eq=wm)["output"]
        actual_kwargs = regime.get_keyword_arguments(
            regime.get_vertex(KnowledgeBase.create)
        )
        expected_kwargs = {
            "linguistic_variables": linguistic_variables,
            "rules": rules,
        }
        assert actual_kwargs == expected_kwargs

        # check that the fuzzy logic rules are added
        assert len(knowledge_base.select_by_tags(tags="rule")) == len(rules)
        assert len(knowledge_base.rules) == len(rules)

        # checking that this query returns the same as the above; they are equivalent
        knowledge_base = regime.graph.vs.find(callable_eq=KnowledgeBase.create)[
            "output"
        ]
        assert len(knowledge_base.select_by_tags(tags="rule")) == len(rules)
        assert len(knowledge_base.rules) == len(rules)

        return knowledge_base
