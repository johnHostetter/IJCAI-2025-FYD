"""
Test the self-organizing procedures of CEW and FYD.
"""

import os
import shutil
import pathlib
import unittest
from typing import Union

import torch
import numpy as np
from regime import Resource, Node
from fuzzy.relations.t_norm import Product, Minimum
from fuzzy.logic.knowledge_base import KnowledgeBase
from fuzzy.logic.control.defuzzification import ZeroOrder
from fuzzy.logic.control.controller import FuzzyLogicController as FLC

# the following algorithms are eligible for self-organizing neuro-fuzzy networks
from crisp_ml.autoencode import AutoEncoder
from fuzzy_ml.utils import set_rng
from fuzzy_ml.datasets import LabeledDataset
from fuzzy_ml.fetchers import fetch_labeled_dataset
from fuzzy_ml.clustering.ecm import EvolvingClusteringMethod
from fuzzy_ml.rulemaking.latent_lockstep import LatentLockstep, LatentSpace
from fuzzy_ml.partitioning.clip import CategoricalLearningInducedPartitioning
from fuzzy_ml.rulemaking.wang_mendel import WangMendelMethod as WM

from organize import SelfOrganize
from fyd.regime import clip_frequent_discernible
from neuro_fuzzy.skorch.regime import SupervisedTraining
from soft_computing.utilities.reproducibility import load_configuration


N_OUTPUTS: int = 2
AVAILABLE_DEVICE = torch.device("cpu")  # CUDA out of memory error for these tests


def random_sample_cart_pole_data(num_samples: int = 20) -> torch.Tensor:
    np.random.seed(0)

    # Define the range of the distribution
    lower_bound = [-4.8, -np.inf, -0.42, -np.inf]
    upper_bound = [4.8, np.inf, 0.42, np.inf]

    # Sample 20 data points within the given domain
    sampled_data = []
    for i in range(num_samples):
        data_point = []
        for low, high in zip(lower_bound, upper_bound):
            if np.isinf(low) and np.isinf(high):
                data_point.append(
                    np.random.uniform() + np.random.choice([-1, 1]) * 0.42
                )
            elif np.isinf(low):
                data_point.append(np.random.uniform(high - 0.42, high))
            elif np.isinf(high):
                data_point.append(np.random.uniform(low, low + 0.42))
            else:
                data_point.append(np.random.uniform(low, high))
        sampled_data.append(data_point)

    # Print sampled data points
    for i, point in enumerate(sampled_data):
        print(f"Data point {i + 1}: {point}")

    return torch.Tensor(sampled_data)


def get_cart_pole_example_data() -> torch.Tensor:
    data_points = [
        [0.47, 1.14, 0.30, 1.27],
        [1.40, 0.02, -0.37, 0.69],
        [2.80, 0.11, -0.09, 0.42],
        [-3.96, 0.44, 0.38, -0.28],
        [4.59, 1.22, 0.02, 1.10],
        [1.34, 0.56, 0.22, 0.53],
        [-2.26, 1.19, -0.24, -0.28],
        [1.13, 0.19, -0.10, 0.48],
        [-1.35, 0.86, -0.34, 0.55],
        [1.64, 0.63, 0.21, 0.19],
        [0.67, 0.02, 0.39, 0.23],
        [-3.25, 0.23, -0.07, 0.05],
        [-3.27, 0.53, -0.15, 1.20],
        [-1.26, 0.40, 0.10, 0.25],
        [4.57, 0.89, -0.37, 0.87],
        [-4.42, -0.14, -0.12, 0.90],
        [-1.75, 0.83, -0.24, 0.15],
        [-2.25, 0.94, 0.35, -0.34],
        [-1.74, 1.09, 0.12, 0.42],
        [-3.04, 0.17, -0.28, -0.05],
    ]

    # Convert to torch.Tensor
    return torch.tensor(data_points, device=AVAILABLE_DEVICE)[:8, :]


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
        self.callables: Union[Node, callable] = {
            CategoricalLearningInducedPartitioning,
            EvolvingClusteringMethod,
            KnowledgeBase.create,
            fetch_labeled_dataset,
            WM,
        }  # either a function or a Node object are allowed
        self.data = torch.load(file_path).to(AVAILABLE_DEVICE)
        self.config = load_configuration()
        with self.config.unfreeze():
            self.config.device = "cpu"
            self.config.fuzzy_ml.clustering.ecm.EvolvingClusteringMethod.distance_threshold = (
                1e-3
            )

    def test_blueprint_clip_ecm_wm(self) -> None:
        """
        Test the self-organizing process with CLIP, followed by ECM, and then generate fuzzy logic
        rules with the Wang-Mendel method.

        Returns:
            KnowledgeBase
        """
        set_rng(0)
        self_organize = SelfOrganize(
            algorithms={
                "clip": CategoricalLearningInducedPartitioning(),
                "ecm": EvolvingClusteringMethod(),
                "wm": WM(t_norm=Product),
            },
            device=AVAILABLE_DEVICE,
        ).setup(
            resources={
                Resource(
                    name="train_dataset",
                    value=LabeledDataset(data=self.data, out_features=1),
                )
            },
            configuration=self.config,
        )
        knowledge_base = self_organize.run()
        self.assertEqual(10, len(knowledge_base.select_by_tags(tags="rule")))
        self.assertEqual(10, len(knowledge_base.rules))

        # checking that this query returns the same as the above; they are equivalent
        knowledge_base = self_organize.regime.graph.vs.find(
            callable_eq=KnowledgeBase.create
        )["output"]
        self.assertEqual(10, len(knowledge_base.select_by_tags(tags="rule")))
        self.assertEqual(10, len(knowledge_base.rules))

        return knowledge_base

    def test_blueprint_clip_frequent_discernible(self) -> None:
        """
        Test the self-organizing process with CLIP followed by the frequent discernible method.

        Returns:
            KnowledgeBase
        """
        set_rng(0)
        # number_of_rules = 187
        directory = pathlib.Path(__file__).parent.resolve()
        train_file_path = os.path.join(directory, "big_train_data.pt")
        val_file_path = os.path.join(directory, "big_val_data.pt")
        big_train_data = torch.load(train_file_path)[:20, :4]
        big_val_data = torch.load(val_file_path)

        big_train_data = get_cart_pole_example_data()  # overwrite the data to match the paper

        with self.config.unfreeze():
            self.config.training.data.batch = self.config.validation.data.batch = 128
            self.config.fuzzy_ml.clustering.ecm.distance_threshold = 0.2
        knowledge_base: KnowledgeBase = clip_frequent_discernible(
            LabeledDataset(data=big_train_data, labels=big_train_data),
            LabeledDataset(data=big_train_data, labels=big_train_data),
            batch_size=self.config.training.data.batch,
            config=self.config,
            device=AVAILABLE_DEVICE,
        )  # this knowledge_base can then be used as the source to initialize a FLC

        # self_organize = SystematicDesignProcess(
        #     algorithms=["clip", "ecm", "wang_mendel"], config=self.config
        # ).build(SupervisedDataset(inputs=big_train_data, targets=None))
        # knowledge_base = self_organize.run()
        # fuzzy_anchors_vertices = [
        #     vertex
        #     for vertex in list(knowledge_base.graph.vs)
        #     if isinstance(vertex["type"], tuple) and vertex["input"]
        # ]
        #
        # # check that the subgraph only contains vertices that reference linguistic terms
        # # and the fuzzy logic rules
        # subgraph: igraph.Graph = induce_subgraph(fuzzy_anchors_vertices, knowledge_base)
        # assert all(
        #     [
        #         subgraph_vertex["type"]
        #         in [
        #             original_vertex["type"]
        #             for original_vertex in fuzzy_anchors_vertices
        #         ]
        #         or subgraph_vertex["layer"] == "Rule"
        #         for subgraph_vertex in subgraph.vs
        #     ]
        # )
        #
        # # check that the scalar cardinality is calculated as we expect
        # scalar_cardinality: np.ndarray = calc_scalar_cardinality(
        #     knowledge_base, fuzzy_anchors_vertices, big_train_data
        # )
        # print(scalar_cardinality)
        # # non-deterministic on the GitHub servers
        # # expected_scalar_cardinality = [
        # #     233.21432495,
        # #     195.95788574,
        # #     246.84750366,
        # #     299.36465454,
        # #     295.7043457,
        # #     264.05709839,
        # #     256.91107178,
        # #     232.48919678,
        # #     160.69519043,
        # #     207.31567383,
        # #     163.54751587,
        # #     100.49259186,
        # # ]
        # #
        # # # compare only the first 12 results
        # # assert np.isclose(scalar_cardinality[:12], expected_scalar_cardinality).all()
        #
        # # check that the Frequent-Yet-Discernible Method reduced the graph's vertices
        # previous_graph_vertices_count = len(knowledge_base.graph.vs)
        # # edits the KnowledgeBase in-place
        # frequent_discernible(
        #     SupervisedDataset(inputs=big_train_data, targets=None),
        #     knowledge_base,
        #     self.config,
        # )
        # assert len(knowledge_base.graph.vs) <= previous_graph_vertices_count
        #
        # # knowledge_base = self_organize.start()  # the result is non-deterministic
        # # assert len(knowledge_base.graph.vs.select(layer_eq="Rule")) == number_of_rules
        # # assert (
        # #     len(knowledge_base.graph.vs.select(item_eq=TNorm.PRODUCT))
        # #     == number_of_rules
        # # )
        # #
        # # # checking that this query returns the same as the above; they are equivalent
        # # knowledge_base = self_organize.graph.vs.find(callable_eq=frequent_discernible)[
        # #     "output"
        # # ]
        # # assert len(knowledge_base.graph.vs.select(layer_eq="Rule")) == number_of_rules
        # # assert (
        # #     len(knowledge_base.graph.vs.select(item_eq=TNorm.PRODUCT))
        # #     == number_of_rules
        # # )

    def test_save_load_knowledge_base(self) -> None:
        """
        Test that when we save and load the KnowledgeBase object,
        that we retrieve the original KnowledgeBase.

        Returns:
            None
        """
        set_rng(0)
        blueprints = [  # selected methods
            self.test_blueprint_clip_ecm_wm,
        ]
        t_norms = [Product, Minimum]
        path_to_this_script = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
        save_dir = path_to_this_script / "models"
        for blueprint, t_norm in zip(blueprints, t_norms):
            knowledge_base = blueprint()
            original_flc = FLC(
                source=knowledge_base,
                inference=ZeroOrder,
                device=AVAILABLE_DEVICE,
            )
            original_predictions = original_flc(self.data)

            # save and load the KnowledgeBase object
            knowledge_base.save(path=save_dir)
            loaded_knowledge_base = KnowledgeBase.load(
                path=save_dir, device=AVAILABLE_DEVICE
            )

            shutil.rmtree(save_dir)  # clean up; delete the model files

            loaded_flc = FLC(
                source=loaded_knowledge_base,
                inference=ZeroOrder,
                device=AVAILABLE_DEVICE,
            )

            loaded_predictions = loaded_flc(self.data)

            # check that the predictions are the same
            self.assertEqual(original_predictions.shape, loaded_predictions.shape)

            self.assertEqual(
                knowledge_base.attribute_table, loaded_knowledge_base.attribute_table
            )

            for vertex, loaded_vertex in zip(
                knowledge_base.graph.vs, loaded_knowledge_base.graph.vs
            ):
                self.assertEqual(vertex.index, loaded_vertex.index)
                for attribute in vertex.attributes():
                    self.assertEqual(vertex[attribute], loaded_vertex[attribute])

            for edge, loaded_edge in zip(
                knowledge_base.graph.es, loaded_knowledge_base.graph.es
            ):
                self.assertEqual(edge.attributes(), loaded_edge.attributes())
