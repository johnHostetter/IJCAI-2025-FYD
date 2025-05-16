# -*- coding: utf-8 -*-
"""
A sample implementation and demo of the proposed Fuzzy Conservative Q-Learning procedure.
The code is not necessarily optimal or efficient with respect to performance, but was more or less
written for interpretability. Some functions that may be difficult to follow, such as ECM,
actually should have a one-to-one correspondence with the original paper's notation
(in the case of ECM, that would be the dynamic-evolving neuro fuzzy system called DENFIS).
"""
import unittest
import warnings
from typing import List
from copy import deepcopy

import torch
import numpy as np
import d3rlpy.algos
import gymnasium as gym
from regime import Resource
from d3rlpy.datasets import get_cartpole
from fuzzy.logic.rule import Rule
from fuzzy.logic.knowledge_base import KnowledgeBase
from fuzzy.logic.variables import LinguisticVariables
from fuzzy.logic.control.defuzzification import ZeroOrder, Mamdani
from fuzzy.logic.control.controller import FuzzyLogicController as FLC
from fuzzy.relations.t_norm import Product, SoftmaxSum

from YACS.yacs import Config
from organize import SelfOrganize
from fuzzy_ml.utils import set_rng
from fuzzy_ml.datasets import LabeledDataset
from fuzzy_ml.rulemaking.wang_mendel import WangMendelMethod
from fuzzy_ml.clustering.ecm import EvolvingClusteringMethod
from fuzzy_ml.partitioning.clip import CategoricalLearningInducedPartitioning

from neuro_fuzzy.xai.explainer import SelectOutputWrapper, SoftExplainer
from neuro_fuzzy.d3rlpy.policy import CustomEncoderFactory
from neuro_fuzzy.d3rlpy.utils import CustomMeanQFunctionFactory
from soft_computing.utilities.performance import performance_boost
from soft_computing.utilities.reproducibility import (
    env_seed,
    load_and_override_default_configuration,
    path_to_project_root,
)
from .toy_examples import toy_mamdani


set_rng(0)
performance_boost()
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# You can set the seed for reproducibility,
# define the number of maximum epochs to allow the agent to train, batch size, etc.
#
# The antecedents or the fuzzy logic rules produced by the `unsupervised` function
# may change due to randomness (there may be more or less, they may behave differently, etc.).
#
# Note: You may need to do a fresh restart of the random number generators,
# environment, and self_organize forth.


class TestSoftExplainer(unittest.TestCase):
    """
    Test the SoftExplainer class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.antecedents, self.consequents, self.rules = toy_mamdani(
            t_norm=Product, device=AVAILABLE_DEVICE
        )
        self.knowledge_base = KnowledgeBase.create(
            linguistic_variables=LinguisticVariables(
                inputs=self.antecedents, targets=self.consequents
            ),
            rules=self.rules,
        )

        self.fuzzy_logic_controller = FLC(
            source=self.knowledge_base,
            inference=Mamdani,
            device=AVAILABLE_DEVICE,
        )

    def test_explain_with_captum(self) -> None:
        """
        Test that explanation generated using the Captum library is functional.

        Returns:
            None
        """
        train_data = torch.tensor(
            [
                [1.5410, -0.2934],
                [-2.1788, 0.5684],
                [-1.0845, -1.3986],
                [0.4033, 0.8380],
                [-0.7193, -0.4033],
            ],
            device=AVAILABLE_DEVICE,
            dtype=torch.float32,
        )
        val_data = torch.tensor(
            [
                [-0.5966, 0.1820],
                [-0.8567, 1.1006],
                [-1.0712, 0.1227],
                [-0.5663, 0.3731],
                [-0.8920, -1.5091],
            ],
            device=AVAILABLE_DEVICE,
            dtype=torch.float32,
        )
        input_data = torch.tensor(
            [
                [1.2, 0.2],
                [1.1, 0.3],
                [2.1, 0.1],
                [2.7, 0.15],
                [1.7, 0.25],
            ],
            device=AVAILABLE_DEVICE,
            dtype=torch.float32,
        )

        temp_agent = SelectOutputWrapper(
            model=self.fuzzy_logic_controller,
            output_index=0,
        )

        explainer = SoftExplainer(
            temp_agent,
            train_data,
            val_data,
        )

        set_rng(0)

        explanation = explainer.explain_with_captum(input_data)
        expected_results = {
            "Integrated Gradients": torch.tensor(
                [
                    [0.4991, 0.0009],
                    [0.1749, -0.0021],
                    [-0.4857, 0.0062],
                    [0.5076, 0.0094],
                    [0.0424, 0.0040],
                ],
                device=AVAILABLE_DEVICE,
                dtype=torch.float32,
            ),
            "Integrated Gradients w/SmoothGrad": torch.tensor(
                [
                    [0.0031, -0.0003],
                    [0.0282, -0.0264],
                    [0.0193, -0.0185],
                    [0.0306, -0.0310],
                    [0.1071, -0.0845],
                ],
                device=AVAILABLE_DEVICE,
                dtype=torch.float32,
            ),  # this one is prone to change after code changes
            "DeepLift": torch.tensor(
                [
                    [2.0975e-08, 2.9527e-10],
                    [3.8015e00, -6.4798e-02],
                    [5.7816e-02, 6.3760e-04],
                    [2.5175e00, 8.3578e-02],
                    [3.7722e-04, 5.3850e-06],
                ],
                device=AVAILABLE_DEVICE,
                dtype=torch.float32,
            ),
            "GradientSHAP": torch.tensor(
                [
                    [0.3000, 0.0370],
                    [0.1534, 0.0207],
                    [-0.5198, 0.0085],
                    [0.1060, -0.0017],
                    [0.0279, 0.1260],
                ],
                device=AVAILABLE_DEVICE,
                dtype=torch.float32,
            ),
            "Feature Ablation": torch.tensor(
                [
                    [5.0000e-01, 1.1060e-01],
                    [1.7280e-01, 2.9543e-02],
                    [2.4473e-03, 6.3481e-04],
                    [2.4864e-01, 8.5491e-02],
                    [1.3655e-05, 6.2480e-06],
                ],
                device=AVAILABLE_DEVICE,
                dtype=torch.float32,
            ),
        }
        for key, values in explanation.items():
            self.assertEqual(values.shape, input_data.shape)
            self.assertTrue(
                torch.allclose(
                    values,
                    expected_results[key],
                    atol=1e-4,
                    rtol=1e-4,
                )
            )

    def test_rules_ignoring_an_input(self) -> None:
        """
        Test the explanation is consistent when attributes are ignored by the
        fuzzy logic controller. In other words, the fuzzy logic controller's rules
        do not have any premises that depend on the ignored attribute.

        Returns:
            None
        """
        rules = self.remove_variable_from_all_rules(variable_index=1)
        knowledge_base = KnowledgeBase.create(
            linguistic_variables=LinguisticVariables(
                inputs=self.antecedents, targets=self.consequents
            ),
            rules=rules,
        )

        train_data = torch.tensor(
            [
                [1.5410, -0.2934],
                [-2.1788, 0.5684],
                [-1.0845, -1.3986],
                [0.4033, 0.8380],
                [-0.7193, -0.4033],
            ],
            device=AVAILABLE_DEVICE,
        ).float()
        val_data = torch.tensor(
            [
                [-0.5966, 0.1820],
                [-0.8567, 1.1006],
                [-1.0712, 0.1227],
                [-0.5663, 0.3731],
                [-0.8920, -1.5091],
            ],
            device=AVAILABLE_DEVICE,
        ).float()
        input_data = torch.tensor(
            [
                [1.2, 0.2],
                [1.1, 0.3],
                [2.1, 0.1],
                [2.7, 0.15],
                [1.7, 0.25],
            ],
            device=AVAILABLE_DEVICE,
        ).float()

        fuzzy_logic_controller = FLC(
            source=knowledge_base,
            inference=Mamdani,
            device=AVAILABLE_DEVICE,
        )

        temp_agent = SelectOutputWrapper(fuzzy_logic_controller, output_index=0)

        explainer = SoftExplainer(
            model=temp_agent,
            train_data=train_data,
            val_data=val_data,
        )

        expected_results = {
            "Integrated Gradients": torch.tensor(
                [
                    [0.5000, 0.0000],
                    [0.1839, 0.0000],
                    [-0.5405, 0.0000],
                    [0.7658, 0.0000],
                    [0.0452, 0.0000],
                ],
                device=AVAILABLE_DEVICE,
            ),
            "Integrated Gradients w/SmoothGrad": torch.tensor(
                [
                    [0.0232, 0.0000],
                    [0.0584, 0.0000],
                    [0.0072, 0.0000],
                    [0.1108, 0.0000],
                    [0.0794, 0.0000],
                ],
                device=AVAILABLE_DEVICE,
                dtype=torch.float32,
            ),  # this one is prone to change after code changes
            "DeepLift": torch.tensor(
                [
                    [3.4673e-08, 0.0000e00],
                    [4.0467e00, 0.0000e00],
                    [1.1963e-01, 0.0000e00],
                    [4.6152e00, 0.0000e00],
                    [5.7169e-04, 0.0000e00],
                ],
                device=AVAILABLE_DEVICE,
            ),
            "GradientSHAP": torch.tensor(
                [
                    [0.3229, 0.0000],
                    [0.2033, 0.0000],
                    [-0.7284, 0.0000],
                    [0.1808, 0.0000],
                    [0.0451, 0.0000],
                ],
                device=AVAILABLE_DEVICE,
            ),
            "Feature Ablation": torch.tensor(
                [
                    [5.0000e-01, 0.0000e00],
                    [1.8394e-01, 0.0000e00],
                    [5.0638e-03, 0.0000e00],
                    [4.5583e-01, 0.0000e00],
                    [2.0695e-05, 0.0000e00],
                ],
                device=AVAILABLE_DEVICE,
            ),
        }

        set_rng(0)

        explanations = explainer.explain_with_captum(input_data)

        set_rng(0)

        explanations_from_visual = explainer.explain_with_captum(
            input_data, visualize=True
        )

        for key, values in explanations.items():
            # SmoothGrad and GradientSHAP are not deterministic, so we can't check for exact
            # equality (i.e., subsequent runs of the same code will not produce the same results)
            self.assertEqual(values.shape, input_data.shape)
            self.assertTrue(
                torch.allclose(
                    values.float(),
                    expected_results[key].float(),
                    atol=1e-4,
                    rtol=1e-4,
                )
            )
            self.assertEqual(explanations_from_visual[key].shape, input_data.shape)
            self.assertTrue(
                torch.allclose(
                    explanations_from_visual[key].float(),
                    expected_results[key].float(),
                    atol=1e-4,
                    rtol=1e-4,
                )
            )

    def remove_variable_from_all_rules(self, variable_index: int) -> List[Rule]:
        """
        Remove the variable from all rules.

        Returns:
            The rules with the variable removed.
        """
        rules = []
        for rule in self.rules:
            current_premise = set(deepcopy(rule.premise.indices[0]))
            for antecedent in rule.premise.indices[0]:
                if (
                    antecedent[0] == variable_index
                ):  # the 0th index is the variable index
                    current_premise.remove(antecedent)
            new_rule = Rule(
                premise=Product(*current_premise, device=rule.premise.device),
                consequence=rule.consequence,
            )
            rules.append(new_rule)
        return rules

    def test_in_missing_values(self) -> None:
        """
        Test the explanation is consistent when values were missing in the input data.

        Returns:
            None
        """
        knowledge_base = KnowledgeBase.create(
            linguistic_variables=LinguisticVariables(
                inputs=self.antecedents, targets=self.consequents
            ),
            rules=self.rules,
        )

        fuzzy_logic_controller = FLC(
            source=knowledge_base,
            inference=Mamdani,
            device=AVAILABLE_DEVICE,
        )

        train_data = torch.tensor(
            [
                [1.5410, -0.2934],
                [-2.1788, 0.5684],
                [-1.0845, -1.3986],
                [0.4033, 0.8380],
                [-0.7193, -0.4033],
            ],
            device=AVAILABLE_DEVICE,
            dtype=torch.float32,
        )
        val_data = torch.tensor(
            [
                [-0.5966, 0.1820],
                [-0.8567, 1.1006],
                [-1.0712, 0.1227],
                [-0.5663, 0.3731],
                [-0.8920, -1.5091],
            ],
            device=AVAILABLE_DEVICE,
            dtype=torch.float32,
        )
        input_data = torch.tensor(
            [
                [np.nan, 1.2],
                [np.nan, 1.0],
            ],
            device=AVAILABLE_DEVICE,
            dtype=torch.float32,
        )

        temp_agent = SelectOutputWrapper(fuzzy_logic_controller, output_index=0)

        explainer = SoftExplainer(
            model=temp_agent,
            train_data=train_data,
            val_data=val_data,
        )

        set_rng(0)

        explanations = explainer.explain_with_captum(input_data)
        expected_results = {
            "Integrated Gradients": torch.tensor(
                [[np.nan, 0.0], [np.nan, 0.0]],
                device=AVAILABLE_DEVICE,
                dtype=torch.float32,
            ),
            "Integrated Gradients w/SmoothGrad": torch.tensor(
                [[np.nan, 0.0], [np.nan, 0.0]],
                device=AVAILABLE_DEVICE,
                dtype=torch.float32,
            ),
            "DeepLift": torch.tensor(
                [[np.nan, 0.0], [np.nan, 0.0]],
                device=AVAILABLE_DEVICE,
                dtype=torch.float32,
            ),
            "GradientSHAP": torch.tensor(
                [[np.nan, 0.0], [np.nan, 0.0]],
                device=AVAILABLE_DEVICE,
                dtype=torch.float32,
            ),
            "Feature Ablation": torch.tensor(
                [[0.0, 0.0], [0.0, 0.0]], device=AVAILABLE_DEVICE, dtype=torch.float32
            ),
        }
        for key, values in explanations.items():
            self.assertEqual(values.shape, input_data.shape)
            self.assertTrue(
                torch.allclose(
                    values.float(),
                    expected_results[key].float(),
                    atol=1e-4,
                    rtol=1e-4,
                    equal_nan=True,
                )
            )

    # def test_explain_with_layer_conductance_with_invalid_model(self) -> None:
    #     """
    #     Test explanations based on the layer conductance method.
    #
    #     Returns:
    #         None
    #     """
    #     train_data = torch.randn((5, 2))
    #     val_data = torch.randn((5, 2))
    #
    #     input_data = torch.tensor(
    #         [
    #             [1.2, 0.2],
    #             [1.1, 0.3],
    #             [2.1, 0.1],
    #             [2.7, 0.15],
    #             [1.7, 0.25],
    #         ]
    #     ).float()
    #
    #     explainer = SoftExplainer(
    #         model=torch.nn.Module(),  # some unrecognized torch.nn.Module
    #         train_data=train_data,
    #         val_data=val_data,
    #     )
    #
    #     self.assertRaises(
    #         NotImplementedError, explainer.explain_with_layer_conductance, input_data
    #     )

    def test_in_offline_reinforcement_learning(self):
        """
        Start the offline reinforcement learning given a configuration and number of episodes to
        use from the training dataset.
        """
        config = load_and_override_default_configuration(
            path_to_project_root()
            / "unit_tests"
            / "test_neuro_fuzzy"
            / "configurations"
            / "cart_pole.yaml"
        )
        # config.merge(env_config, exclusive=False)
        if config.output.verbose:
            config.print(ignored_keys=())

        with config.unfreeze():
            config.environment = Config({"name": "CartPole-v1"})
        print(
            f"Using seed {config.reproducibility.seed} to solve: {config.reproducibility.seed}"
        )
        # https://github.com/openai/gym/issues/2540
        # https://github.com/openai/gym/pull/2671 (merged)
        # change render_mode to 'human' for display, else None
        env = gym.make(config.environment.name)
        env_seed(env, config.reproducibility.seed)

        # Number of states and number of actions
        n_state, n_action = env.observation_space.shape[0], env.action_space.n

        # apply Fuzzy Conservative Q-Learning algorithm to learn the Q-values
        replay_buffer, _ = get_cartpole(
            dataset_type="replay"
        )  # replay_buffer is d3rlpy.dataset.replay_buffer.ReplayBuffer

        observations = torch.tensor(
            [
                observation
                for episode in replay_buffer.episodes
                for observation in episode.observations
            ]
        )
        train_observations = observations[:200]
        val_observations = observations[-100:]

        knowledge_base: KnowledgeBase = (
            SelfOrganize(
                algorithms={
                    "clip": CategoricalLearningInducedPartitioning(),
                    "ecm": EvolvingClusteringMethod(),
                    "wm": WangMendelMethod(t_norm=Product),
                },
                device=AVAILABLE_DEVICE,
            )
            .setup(
                resources={
                    Resource(
                        name="train_dataset",
                        value=LabeledDataset(
                            data=train_observations, out_features=n_action
                        ),
                    )
                },
                configuration=config,
            )
            .run()
        )

        encoder_factory = CustomEncoderFactory(
            source=knowledge_base,
            inference=ZeroOrder,
            t_norm=SoftmaxSum,
            device=AVAILABLE_DEVICE,
        )
        algorithm = d3rlpy.algos.DiscreteCQLConfig(
            learning_rate=1e-3,
            encoder_factory=encoder_factory,
            # q_func_factory=CustomMeanQFunctionFactory(share_encoder=False),
        ).create(device="cuda" if torch.cuda.is_available() else "cpu")
        # evaluators = {"environment": d3rlpy.metrics.EnvironmentEvaluator(env)}
        algorithm.fit(
            replay_buffer,
            # eval_episodes=val_episodes,
            n_steps=1,  # very low for testing purposes
            n_steps_per_epoch=1,  # very low for testing purposes
            # n_epochs=config.training.epochs,
            # evaluators=evaluators,
        )
        score = d3rlpy.metrics.EnvironmentEvaluator(
            env,
            n_trials=10,
        )(algorithm, dataset=None)
        print(f"The average reward was {score}.")

        # explain the agent's behavior
        # def explain_agent(algorithm):
        #     return lambda x: algorithm._impl.q_function(x)[:, 0]
        input_data = train_observations[:1]

        # explainer = SoftExplainer(
        #     model=algorithm._impl.q_function[0]._encoder.flc,
        #     train_data=train_observations,
        #     val_data=val_observations,
        # )
        # explainer.explain_with_layer_conductance(input_data)

        temp_agent = SelectOutputWrapper(
            algorithm._impl.q_function[0]._encoder.flc, output_index=0
        )

        explainer = SoftExplainer(
            model=temp_agent,  # ._q_funcs[0]._encoder.flc,
            train_data=train_observations,
            val_data=val_observations,
        )

        # explainer.explain_with_shap_values(input_data)
        explainer.explain_with_captum(input_data)
        explainer.explain_with_counterfactuals(
            input_data, output_file_name="counterfactuals"
        )  # handle UserConfigValidationException
