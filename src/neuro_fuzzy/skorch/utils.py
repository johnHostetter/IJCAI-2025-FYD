"""
This file provides extension classes for the skorch library.
"""

from skorch.callbacks import EpochScoring

import wandb


class WandbEpochScoring(EpochScoring):
    """
    Overrides the behavior of recording the score to also record it to wandb.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _record_score(self, history, current_score):
        """Record the current store and, if applicable, if it's the best score
        yet.

        """
        history.record(self.name_, current_score)
        wandb.log({self.name_: current_score})

        is_best = self._is_best_score(current_score)
        if is_best is None:
            return

        history.record(self.name_ + "_best", bool(is_best))
        if is_best:
            self.best_score_ = (  # pylint: disable=attribute-defined-outside-init
                current_score
            )
