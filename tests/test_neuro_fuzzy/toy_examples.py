"""
Functions related to toy examples for the Neuro-Fuzzy module.
"""

from typing import Tuple, List, Type

import torch
import numpy as np
from fuzzy.logic.rule import Rule
from fuzzy.sets.impl import Gaussian
from fuzzy.relations.t_norm import TNorm
from fuzzy.relations.n_ary import NAryRelation


AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def toy_mamdani(
    t_norm: Type[TNorm],
    device: torch.device,
) -> Tuple[List[Gaussian], List[Gaussian], List[Rule]]:
    """
    A toy example for defining some Mamdani Fuzzy Logic Controller.

    Args:
        t_norm: The t-norm to use for the fuzzy logic rules.
        device: The device to use for the PyTorch objects.

    Returns:
        A list of antecedents, a list of consequents, and a list of rules.
    """
    premises = [
        Gaussian(
            centers=np.array([1.2, 3.0, 5.0, 7.0]),
            widths=np.array([0.1, 0.4, 0.6, 0.8]),
            device=device,
        ),
        Gaussian(
            centers=np.array([0.2, 0.6, 0.9, 1.2]),
            widths=np.array([0.4, 0.4, 0.5, 0.45]),
            device=device,
        ),
    ]
    consequences = [
        Gaussian(
            centers=np.array([0.5, 0.3]),
            widths=np.array([0.1, 0.4]),
            device=device,
        ),
        Gaussian(
            centers=np.array([-0.2, -0.7, -0.9]),
            widths=np.array([0.4, 0.4, 0.5]),
            device=device,
        ),
    ]
    rules = [
        Rule(
            premise=t_norm((0, 0), (1, 0), device=device),
            consequence=NAryRelation((0, 0), (1, 1), device=device),
            # used to have to start counting from input variables' size
            # consequence=NAryRelation((2, 0), (3, 1), device=device),
        ),
        Rule(
            premise=t_norm((0, 1), (1, 0), device=device),
            consequence=NAryRelation((0, 1), (1, 2), device=device),
            # consequence=NAryRelation((2, 1), (3, 2), device=device),
        ),
        Rule(
            premise=t_norm((0, 1), (1, 1), device=device),
            consequence=NAryRelation((0, 0), (1, 0), device=device),
            # consequence=NAryRelation((2, 0), (3, 0), device=device),
        ),
    ]
    return premises, consequences, rules
