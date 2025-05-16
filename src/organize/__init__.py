"""
Implements functions and classes responsible for the design of soft computing solutions,
such as the SelfOrganize class. The SelfOrganize class is a regime that facilitates the
design of self-organizing solutions by adding callable function references to a KnowledgeBase
graph as vertices, and using the edges that connect the vertices (i.e., functions) to determine
the flow of data/information from function to function.

The SelfOrganize class is used to provide a convenient interface for the design of self-organizing
neuro-fuzzy systems. The class contains an instance of a Regime, which provides the functionality
to manage the execution of functions in a thread-safe manner. The SelfOrganize class adds the
ability to create a KnowledgeBase graph, which is used to represent the flow of data between
functions in the self-organizing system.
"""

from typing import Set, OrderedDict, List, Tuple, Any, Dict

import torch
from regime import Regime, Resource, Node
from fuzzy.logic.knowledge_base import KnowledgeBase

from fuzzy_ml.fetchers import fetch_labeled_dataset


class SelfOrganize:
    """
    The SelfOrganize class facilitates the convenient design of a self-organizing KnowledgeBase
    by adding callable function references to a Regime graph as vertices, and using the edges that
    connect the vertices (i.e., functions) to determine the flow of data/information from function
    to function.
    """

    def __init__(self, algorithms: Dict[str, Node], device: torch.device):
        """
        Initializes the SelfOrganize class. The device is used to determine where the
        computation will be executed. It is required upon initialization to ensure that
        the SelfOrganize process is properly set up.

        Args:
            algorithms: A dictionary mapping a unique ID to each algorithm to use.
            device: The device to use.
        """
        super().__init__()
        self.__algorithms_dict = algorithms
        self.algorithms = set(algorithms.values()) | {
            fetch_labeled_dataset,
            KnowledgeBase.create,
        }  # add helpful wrapper functions used often in self-organizing systems
        self.device = device
        self.regime = Regime(
            callables=self.algorithms,
            resources={Resource(name="device", value=self.device)},
        )

    def __getitem__(self, key: str) -> Node:
        return self.__algorithms_dict[key]

    def get_regime_edges(self) -> List[Tuple[Any, Any, int]]:
        """
        Get the edges to be added to the Regime object.

        Returns:
            The edges to be added to the Regime object.
        """
        edges = []
        make_knowledge_base_required: bool = True
        for algorithm in self.algorithms:
            if isinstance(algorithm, Node):
                edges.extend(algorithm.edges())
                if algorithm.resource_name == "knowledge_base":
                    make_knowledge_base_required = False

        if make_knowledge_base_required:
            edges.extend(
                [
                    ("linguistic_variables", KnowledgeBase.create, 0),
                    ("rules", KnowledgeBase.create, 1),
                ]
            )

        return edges

    def setup(
        self, resources: Set[Resource], configuration: OrderedDict
    ) -> "SelfOrganize":
        """
        Set up the SelfOrganize object for execution. The configuration comes *after* the algorithms
        are determined to ensure that all required hyperparameters are available/defined. Resources
        can be added to the underlying Regime at any time; however, it is recommended to do so
        during the setup phase for SelfOrganize, to keep the code consistent, but also to allow
        the given resources to be flexible and changeable (i.e., a SelfOrganize process never
        changes, but what it is applied to will).

        Note: Although in Regime.setup() the clean_up parameter is set to False, in
        SelfOrganize.setup() it is set to True to ensure that isolated processes (such as unused
        resources/functions) are removed for clarity. Also, some optional arguments for
        Regime.setup() are now required to ensure that the SelfOrganize process is properly set up.

        Args:
            resources: The resources to use.
            configuration: The configuration to use.

        """
        self.regime.setup(
            configuration=configuration,
            edges=self.get_regime_edges(),
            resources=resources,
            clean_up=True,
        )
        return self

    def run(self) -> KnowledgeBase:
        """
        Run the SelfOrganize process (i.e., essentially a restricted Regime call).

        Returns:
            The resulting KnowledgeBase.
        """
        # dict mapping process/resource names to their output
        results: Dict[str, Any] = self.regime.start()

        # knowledge base's key is the module and function name of 'KnowledgeBase.create'
        access_key: str = (
            f"{KnowledgeBase.create.__module__}.{KnowledgeBase.create.__name__}"
        )

        if access_key in results:
            return results[
                access_key
            ]  # a resource produced by KnowledgeBase.create (default)
        if "knowledge_base" in results:
            return results[
                "knowledge_base"
            ]  # a resource produced by a Node (e.g. FTARM)

        raise ValueError("KnowledgeBase not found in results.")
