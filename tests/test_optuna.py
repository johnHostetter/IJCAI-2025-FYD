import pickle
import unittest
from typing import List

import torch
import optuna

from experiments.supervised.self_organize import (
    build_and_train_neuro_fuzzy_network_factory,
)
from soft_computing.utilities.reproducibility import path_to_project_root


AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestOptuna(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler = optuna.samplers.NSGAIISampler()
        self.study = optuna.create_study(
            sampler=self.sampler, directions=["minimize", "minimize"]
        )

        self.algorithm_names: List[str] = ["clip_ecm_wm", "clip_fyd", "latent_lockstep"]

    def test_kin8nm(self) -> None:
        """
        Run the experiments for the kin8nm dataset. Check that the output is saved in the
        appropriate directory.

        Returns:

        """
        dataset_name, data_id = "kin8nm", 189
        # dataset_name, data_id = "puma32H", 308
        for algorithm_name in self.algorithm_names:
            output_directory = (
                path_to_project_root()
                / "data"
                / "output"
                / "skorch"
                / dataset_name
                / algorithm_name
            )
            self.study.optimize(
                build_and_train_neuro_fuzzy_network_factory(
                    dataset_name=dataset_name,
                    data_id=data_id,
                    method_name=algorithm_name,
                    device=AVAILABLE_DEVICE,
                    output_directory=output_directory,
                ),
                n_trials=3,
            )
            with open(str(output_directory / "optuna"), "ab") as outfile:
                pickle.dump(self.study, outfile)

            # check that the output directory was created, if it did not exist
            self.assertTrue(output_directory.exists())
            # check that the output directory is not empty
            self.assertTrue(any((output_directory.iterdir())))
            # check that the optuna file was created
            self.assertTrue((output_directory / "optuna").exists())
