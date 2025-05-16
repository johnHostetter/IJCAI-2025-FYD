"""
This script implements the necessary operations for the FYD method to self-organize
neuro-fuzzy networks.
"""

from typing import List, OrderedDict

import torch
from torch.utils.data import DataLoader
from fuzzy.relations.t_norm import Product
from fuzzy.sets import Membership
from fuzzy.logic.rule import Rule
from fuzzy.logic.knowledge_base import KnowledgeBase
from fuzzy.logic.variables import LinguisticVariables
from fuzzy.logic.control.defuzzification import Mamdani
from fuzzy.logic.control.controller import FuzzyLogicController
from fuzzy_ml.rulemaking.wang_mendel import WangMendelMethod as WM
from fuzzy_ml.clustering.ecm import EvolvingClusteringMethod as ECM
from fuzzy_ml.partitioning.clip import CategoricalLearningInducedPartitioning as CLIP
from fuzzy_ml.datasets import LabeledDataset
from regime import Resource

from organize import SelfOrganize
from fyd.heuristic import frequent_discernible
from neuro_fuzzy.skorch.regime import SupervisedTraining


def clip_frequent_discernible(
    training_data,
    validation_data,
    batch_size: int,
    config: OrderedDict,
    device: torch.device,
) -> KnowledgeBase:
    """
    The CLIP-Frequent-Discernible approach to self-organize FLCs.

    The idea behind this method is to find the antecedents that make
    each fuzzy logic rule discernible from one another (Rough Sets)
    while also taking into account their frequency, or in other words,
    their support across the entire data. We want antecedents in our
    fuzzy logic rules that discern themselves well, but are
    well-supported or activated by the input data.

    Note: At some point during this self organization process, the code
    will automatically switch to a GPU if one is available, regardless
    of user-defined configuration settings.

    Args:
        training_data: The training data to use.
        validation_data: The validation data to use.
        batch_size: The batch size to use.
        config: The configuration to use.
        device: The device to use.

    Returns:
        An object to self organize a neuro-fuzzy network according to the CLIP-FYD approach.
    """

    def make_mamdani_flc(
        linguistic_variables: LinguisticVariables,
        rules: List[Rule],
    ) -> FuzzyLogicController:
        """
        Given the linguistic variables and rules, make a Mamdani FuzzyLogicController.

        Args:
            linguistic_variables: The linguistic variables used in the given rules; only the
            antecedents are needed here, though.
            rules: The rules used to make the FuzzyLogicController.

        Returns:
            A Mamdani FuzzyLogicController.
        """
        mamdani_rules = []
        for rule in rules:
            # make the rules' premises reference themselves to adjust their parameters
            mamdani_rules.append(
                Rule(
                    premise=rule.premise,
                    consequence=rule.premise,
                )
            )
        # change the linguistic variables for the output space to be the same as the input space
        mamdani_linguistic_variables: LinguisticVariables = LinguisticVariables(
            inputs=linguistic_variables.inputs,
            targets=linguistic_variables.inputs,
        )
        knowledge_base = KnowledgeBase.create(
            linguistic_variables=mamdani_linguistic_variables,
            rules=mamdani_rules,
        )
        # the same linguistic variables must be detectable in the target space
        # knowledge_base.set_granules(
        #     nodes=linguistic_variables.inputs, source="expert", is_input=False
        # )
        # knowledge_base.add_fuzzy_granules(
        #     linguistic_variables.inputs, is_input=False
        # )  # register granules' anchors (for rules)
        # add_stacked_granule_helper(knowledge_base, is_input=False)
        return FuzzyLogicController(
            source=knowledge_base,
            inference=Mamdani,
            device=device,
        )

    def prepare_mamdani_flc_target(
        labeled_dataset: LabeledDataset,
        fuzzy_logic_controller: FuzzyLogicController,
        batch_size: int,
    ) -> LabeledDataset:
        """
        Given the input data, calculates the degree to which the rules should be
        activated per data observation, and appends this information to each data point.

        This function is used to train the Mamdani FuzzyLogicController encoder, which helps refine
        fuzzy sets defined in the input space.

        Args:
            labeled_dataset: The input data.
            fuzzy_logic_controller: The FuzzyLogicController to use.
            batch_size: The batch size to use.

        Returns:
            A Dataset mapping the input data to both the input data and the rule activations.
        """
        print("calc rules")
        data_loader = DataLoader(
            labeled_dataset.data,
            batch_size=batch_size,
            shuffle=False,
        )
        rule_activations = []
        for samples in data_loader:
            sample_rule_activations: Membership = fuzzy_logic_controller.engine(
                fuzzy_logic_controller.input(samples.to(device))
            )
            rule_activations.append(sample_rule_activations.degrees.detach())
        rule_activations = torch.vstack(rule_activations)
        print("done")

        return LabeledDataset(
            data=labeled_dataset.data,
            # auto-encoding; so the target is the input data + fuzzy rule activations
            labels=torch.hstack((labeled_dataset.data.to(device), rule_activations)),
        )

    class CalculateRuleActivation(torch.nn.Module):
        """
        This class will produce a horizontal stack of both the
        Fuzzy Logic Controller's inferred output, and its rule activations.
        """

        def __init__(self, model, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model = model

        def forward(self, input_data) -> torch.Tensor:
            """
            Calculates the output of the FuzzyLogicController and its rule activations.
            """
            return torch.hstack(
                (
                    self.model(input_data),
                    self.model.engine(self.model.input(input_data)).degrees,
                )
            )

    self_organize = SelfOrganize(
        algorithms={
            "clip": CLIP(),
            "ecm": ECM(),
            "wm": WM(t_norm=Product),
        },
        device=device,
    ).setup(
        resources={
            Resource(
                name="train_dataset",
                value=training_data,
            )
        },
        configuration=config,
    )

    _: KnowledgeBase = self_organize.run()  # run the initial processes

    # search through the Regime graph to find the linguistic variables and rules
    linguistic_variables: LinguisticVariables = self_organize.regime.graph.vs.find(
        name_eq="linguistic_variables"
    )["output"]
    rules: List[Rule] = self_organize.regime.graph.vs.find(name_eq="rules")["output"]

    mamdani_flc = make_mamdani_flc(linguistic_variables, rules)
    mamdani_train_data = prepare_mamdani_flc_target(
        training_data, mamdani_flc, batch_size
    )
    mamdani_val_data = prepare_mamdani_flc_target(
        validation_data, mamdani_flc, batch_size
    )

    # SupervisedTraining returns a NeuralNetRegressor, so we need to get the model from the
    # NeuralNetRegressor's module (CalculateRuleActivation) to get the trained premises
    trained_mamdani_flc = SupervisedTraining(
        learning_rate=1e-4,
        max_epochs=12,
        batch_size=batch_size,
        patience=4,
        monitor="valid_loss",
    )(
        CalculateRuleActivation(mamdani_flc),
        mamdani_train_data,
        mamdani_val_data,
        device=device,
    ).module.model  # VERY IMPORTANT: get the model from the NeuralNetRegressor

    # make the kb
    kb = KnowledgeBase.create(
        linguistic_variables=trained_mamdani_flc.linguistic_variables(),
        rules=rules,
    )

    # reduce the kb (FYD)
    reduced_kb = frequent_discernible(
        training_data,
        kb,
        device=device,
    )

    return reduced_kb
