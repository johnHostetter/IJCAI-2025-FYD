"""
Test the SoftRegressor class, which is a wrapper around the FuzzyLogicController class to
train a neural network using the skorch library.
"""

import pickle
import unittest
from typing import List
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from fuzzy.sets import LogGaussian, FuzzySetGroup
from fuzzy.relations.t_norm import SoftmaxSum
from fuzzy.logic.knowledge_base import KnowledgeBase
from fuzzy.logic.control.configurations.data import Shape
from fuzzy.logic.control.defuzzification import ZeroOrder, TSK, Mamdani
from fuzzy.logic.control.controller import FuzzyLogicController as FLC
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from skorch.callbacks import EarlyStopping, Checkpoint, GradientNormClipping
from fuzzy_ml import LabeledGaussian
from fuzzy_ml.datasets import (
    RegressionDatasets,
    RegressionDatasetConfig,
    LabeledDataset,
    convert_dat_to_csv,
)

from YACS.yacs import Config
from gumbel.morphism import Morphism
from gumbel.specifications import Specifications
from gumbel.t_norm import SoftmaxLayerNorm
from organize.wrappers.clustering import g_means_wrapper
from neuro_fuzzy.skorch.policy import SoftRegressor
from neuro_fuzzy.skorch.utils import WandbEpochScoring
from soft_computing.utilities.reproducibility import (
    load_configuration,
    path_to_project_root,
)

import wandb


AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def perform_kfold_cross_validation(
    dataset_config: RegressionDatasetConfig,
    n_splits: int,
    project_name: str,
    group_name: str,
) -> None:
    """
    Perform k-fold cross-validation on the given dataset.

    Args:
        dataset_config: The dataset configuration.
        n_splits: The number of splits.
        project_name: The project name for wandb.
        group_name: The group name for wandb.

    Returns:
        None
    """
    input_features = get_features(dataset_config)
    target_features = dataset_config.target_names

    k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    for idx, (train_index, test_index) in enumerate(k_fold.split(dataset_config.frame)):
        train_data = dataset_config.frame.iloc[train_index]
        test_data = dataset_config.frame.iloc[test_index]
        # reinitialize wandb for each fold
        wandb.init(
            # set the wandb project where this run will be logged
            project=project_name,
            # specify a group to organize individual runs
            group=group_name,
            # the name of this run
            name=f"{dataset_config.name}_{idx}",
            # allows multiple .init calls in the same process
            reinit=True,
            # track hyperparameters and run metadata
            config={
                "learning_rate": 0.02,
                "architecture": "NFN",
                "dataset": dataset_config.name,
                "epochs": 10,
            },
        )
        _ = apply_pipeline_and_train(
            train_data,
            test_data,
            input_features,
            target_features,
            output_filename=dataset_config.name + f"_{idx}",
        )


def predefined_split(directory: Path, project_name: str, group_name: str) -> None:
    """
    The data has already been split into training and testing sets.

    Args:
        directory: The directory containing the training and testing sets.
        project_name: The project name for wandb.
        group_name: The group name for wandb.

    Returns:
        None
    """
    train_dat_files: List[Path] = sorted(list(directory.glob("*-5-*tra.dat")))
    test_dat_files: List[Path] = sorted(list(directory.glob("*-5-*tst.dat")))
    idx: int = 0
    if len(train_dat_files) == 0 or len(test_dat_files) == 0:
        raise FileNotFoundError(
            "No training or testing files found; check if unzip is required."
        )
    if len(train_dat_files) != len(test_dat_files):
        raise ValueError(
            "The number of training and testing files are not equal. "
            "Please check the data directory."
        )
    for train_dat_file, test_dat_file in zip(train_dat_files, test_dat_files):
        print(f"Train file: {train_dat_file.name}")
        print(f"Test file: {test_dat_file.name}")
        train_dataset_config: RegressionDatasetConfig = convert_dat_to_csv(
            train_dat_file
        )
        test_dataset_config: RegressionDatasetConfig = convert_dat_to_csv(test_dat_file)
        input_features = get_features(train_dataset_config)
        assert (
            train_dataset_config.target_names == test_dataset_config.target_names,
            "The target names are different between the train and test data.",
        )
        target_features = train_dataset_config.target_names
        train_data, test_data = train_dataset_config.frame, test_dataset_config.frame
        # reinitialize wandb for each fold
        wandb.init(
            # set the wandb project where this run will be logged
            project=project_name,
            # specify a group to organize individual runs
            group=group_name,
            # the name of this run
            name=f"{train_dataset_config.name}_{idx}",
            # allows multiple .init calls in the same process
            reinit=True,
            # track hyperparameters and run metadata
            config={
                "learning_rate": 0.02,
                "architecture": "NFN",
                "dataset": train_dataset_config.name,
                "epochs": 10,
            },
        )
        _ = apply_pipeline_and_train(
            train_data,
            test_data,
            input_features,
            target_features,
            output_filename=train_dat_file.name.replace("tra.dat", ""),
        )
        idx += 1


def get_features(dataset_config: RegressionDatasetConfig) -> List[str]:
    """
    Given a dataset configuration, return the feature names (after eliminating some due to rules).
    These features are dropped because they have a low number of unique values, which performs
    poorly with fuzzy logic.

    Args:
        dataset_config: The dataset configuration.

    Returns:
        The feature names.
    """
    filter_features = dataset_config.frame.nunique() <= 10
    removed_features = list(filter_features[filter_features].index)
    return list(
        filter(lambda i: i not in removed_features, dataset_config.feature_names)
    )


def apply_pipeline_and_train(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    input_features: List[str],
    target_features: List[str],
    output_filename: str,
) -> float:
    """
    Apply the pipeline and train the model. Save the history and knowledge base.

    Args:
        train_data: The training data.
        test_data: The testing data.
        input_features: The input features.
        target_features: The target features.
        output_filename: The output filename.

    Returns:
        The mean squared error on the test dataset.
    """
    train_input, test_input = (
        np.nan_to_num(train_data[input_features].values),
        np.nan_to_num(test_data[input_features].values),
    )
    train_output: np.ndarray = np.nan_to_num(
        train_data[target_features].to_numpy().astype(float)
    )
    test_output: np.ndarray = np.nan_to_num(
        test_data[target_features].to_numpy().astype(float)
    )

    num_of_selected_features: int = len(input_features)
    pipeline = Pipeline(
        [
            (
                "scale",
                FeatureUnion(
                    [
                        ("minmax", MinMaxScaler()),
                        ("normalize", Normalizer()),
                        ("standardize", StandardScaler()),
                    ]
                ),
            ),
            (
                "select",
                SelectKBest(k=num_of_selected_features),
            ),  # keep input size constant
            # ("net", nn_reg),  # can also have the model come after the feature selection
        ]
    )
    pipeline.fit(train_input, train_output)
    config: Config = load_configuration()

    print("Finding the number of rules...")
    normalizer = Normalizer()
    normalized_train_data: np.ndarray = normalizer.fit_transform(train_input)
    labeled_clusters: LabeledGaussian = g_means_wrapper(
        LabeledDataset(data=torch.tensor(normalized_train_data), out_features=1),
        device=AVAILABLE_DEVICE,
    )
    number_of_rules: int = min(labeled_clusters.get_centers().shape[0], 10)
    print(f"Number of rules: {number_of_rules}")

    static = True
    flc = FLC(
        source=Specifications(
            shape=Shape(num_of_selected_features, 5, number_of_rules, 1, 1),
            fuzzy_set=LogGaussian,  # old
            t_norm=SoftmaxLayerNorm,  # old: SoftmaxSum
            mamdani=False,
            device=AVAILABLE_DEVICE,
            fuzzy_set_group=FuzzySetGroup if static else Morphism,
            configuration={
                "architecture": {
                    "init_width": 0.1,
                    "aggregation": "sum",
                    "epsilon": 0.1,
                    "add_premise_delay": 1,
                    "epsilon_filter": 0.0,
                    "gumbel_temperature": 1.0,
                    "premise_sampling": "straight_through",
                    "noise_delay": 1,
                    "rule_weights": True,
                    "layer_normalization": True,
                    "premise_activation": "softmax",
                },
            },
        ),
        inference=TSK,
        disabled_parameters=None,
        device=AVAILABLE_DEVICE,
    )
    flc.train()
    monitor = lambda model: all(
        model.history[-1, ("train_loss_best", "valid_loss_best")]
    )
    checkpoint_dir: Path = path_to_project_root() / "output" / "skorch"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    soft_regressor = SoftRegressor(
        flc,
        lr=3e-2,
        optimizer=torch.optim.Adam,
        criterion=torch.nn.MSELoss,
        max_epochs=100,
        batch_size=int(config.training.data.batch),
        device="cuda" if torch.cuda.is_available() else "cpu",
        callbacks=[
            EarlyStopping(patience=10, monitor="valid_loss"),  # , load_best=True),
            GradientNormClipping(1.0),
            Checkpoint(monitor=monitor, dirname=checkpoint_dir, load_best=True),
            # valid metrics here: https://scikit-learn.org/stable/modules/model_evaluation.html
            WandbEpochScoring(scoring="neg_mean_squared_error", lower_is_better=True),
        ],
    )
    soft_regressor.fit(
        torch.tensor(pipeline.transform(train_input), dtype=torch.float32),
        torch.tensor(train_output, dtype=torch.float32),
    )
    soft_regressor.module.eval()
    predictions = soft_regressor.predict(
        torch.tensor(pipeline.transform(test_input), dtype=torch.float32)
    )
    test_loss = mean_squared_error(test_output, predictions)
    print(f"Test MSE: {test_loss}")

    history = soft_regressor.history
    history_df = pd.DataFrame(history)
    history_df["test_loss"] = test_loss
    history_df["num_of_rules"] = number_of_rules
    history_df["num_of_terms"] = soft_regressor.module.input.centers.shape[-1]
    del history_df["batches"]
    history[-1]["test_loss"] = test_loss

    history_file = path_to_project_root() / "data" / "history"
    history_file.mkdir(parents=True, exist_ok=True)
    history_file = history_file / f"{output_filename}.pickle"
    with open(history_file, "wb") as out_file:
        pickle.dump(history, out_file)

    history_df.to_csv(
        history_file.parent / f"{output_filename}.csv",
        index=False,
    )

    if isinstance(flc.source, KnowledgeBase):
        kb_dir = (
            path_to_project_root() / "output" / "knowledge_base" / f"{output_filename}"
        )
        kb_dir.mkdir(parents=True, exist_ok=True)
        flc.source.save(kb_dir)

    return test_loss


class TestSkorch(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.regression_datasets: RegressionDatasets = RegressionDatasets()
        self.project_name: str = "test-skorch"
        self.output_directory: Path = (
            path_to_project_root() / "data" / "output" / "skorch" / self.project_name
        )

    def test_load_regression_data(self) -> None:
        """
        Test the get_regression_data function.

        Returns:
            None
        """
        expected_shapes = {
            "autoMPG6": (392, 6),
            "delta_elv": (9517, 7),
            "forestFires": (517, 13),
            "friedman": (1200, 6),
            "mv": (40768, 11),
            "plastic": (1650, 3),
            "house_16H": (22784, 17),
            "Ailerons": (13750, 41),
            "treasury": (1049, 16),
            "weather_izmir": (1461, 10),
            "california_housing": (20640, 10),
            "pumadyn32nh": (8192, 33),
            "pol": (15000, 27),
        }
        for name, dataset_config in self.regression_datasets.datasets.items():
            self.assertEqual(dataset_config.frame.shape, expected_shapes[name])

    def test_soft_regressor(self) -> None:
        """
        Test the SoftRegressor class.

        Returns:
            None
        """
        num_splits: int = 2
        perform_kfold_cross_validation(
            self.regression_datasets["autoMPG6"],
            n_splits=num_splits,
            project_name=self.project_name,
            group_name=f"{num_splits}-fold-autoMPG6",
        )

    def test_existing_keel_splits(self) -> None:
        """
        Test the predefined_split function.

        Returns:
            None
        """
        keel_data_directory = path_to_project_root() / "data" / "keel" / "regression"
        if keel_data_directory.exists():
            # only run this unit test if we have the data
            directories: List[Path] = list(keel_data_directory.glob("*"))

            for directory in directories[:2]:  # only use the first two datasets
                print(f"Data directory: {directory.name}")
                predefined_split(
                    directory=directory,
                    project_name=self.project_name,
                    group_name="predefined-split",
                )


if __name__ == "__main__":
    TestSkorch().test_load_regression_data()
