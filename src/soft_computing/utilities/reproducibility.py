"""
Provides functions that help guarantee reproducibility.
"""

import pathlib
from typing import Union

import torch
from fuzzy_ml.utils import set_rng

from YACS.yacs import Config


def env_seed(env, seed: int) -> None:
    """
    Set the random number generator, and also set the random number generator for gym.env.

    Args:
        env: The environment.
        seed: The seed to use for the random number generator.

    Returns:
        None
    """
    set_rng(seed)
    env.reset(seed=seed)  # for older version of gym (e.g., 0.21) use env.seed(seed)
    env.action_space.seed(seed)


def path_to_project_root() -> pathlib.Path:
    """
    Return the path to the root of the project.

    Returns:
        The path to the root of the project.
    """
    return pathlib.Path(__file__).parent.parent.parent.parent


def load_configuration(
    file_name: Union[str, pathlib.Path] = "default_configuration.yaml",
) -> Config:
    """
    Load and return the default configuration that should be used for models, if another
    overriding configuration is not used in its place.

    Args:
        file_name: Union[str, pathlib.Path] Either a file name (str) where the function will look up
        the *.yml configuration file on the parent directory (i.e., git repository) level, or a
        pathlib.Path where the object redirects the function to a specific location that may be in
        a subdirectory of this repository.

    Returns:
        The configuration settings.
    """
    file_path = path_to_project_root() / "configurations" / file_name
    config = Config(str(file_path))
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    with config.unfreeze():
        # parse through the configuration settings and convert them to their true values
        config.training.learning_rate = float(config.training.learning_rate)
    return config


def load_and_override_default_configuration(path: pathlib.Path) -> Config:
    """
    Load the default configuration file and override it with the configuration file given by
    'path'. This function is useful for when you want to override the default configuration
    settings with a configuration file that is not the default configuration file. For example,
    you may want to override the default configuration settings with the configuration settings
    for a specific experiment.

    Args:
        path: A file path to the configuration file that should be merged
        with the default configuration.

    Returns:
        The custom configuration settings.
    """
    # the default configuration
    configuration = load_configuration()
    # the custom configuration
    custom_configuration = load_configuration(path)
    configuration.merge(custom_configuration, exclusive=False)
    # if configuration.output.verbose:
    #     configuration.print(ignored_keys=())
    return configuration
