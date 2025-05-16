"""
Implements supervised training and can be used for auto-encoder training as well.
"""

import torch
from regime import Node
from skorch import NeuralNetRegressor  # to make PyTorch model compatible with sklearn
from skorch.helper import predefined_split
from skorch.callbacks import EarlyStopping

from fuzzy_ml.datasets import LabeledDataset


class SupervisedTraining(Node):
    """
    A common metaclass for supervised training of a model given a labeled dataset.
    """

    def __init__(
        self,
        loss: torch.nn.Module = torch.nn.MSELoss,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        resource_name: str = "model",
        *args,
        **kwargs,
    ):
        """
        Initialize the supervised training algorithm with the model, device, loss function, and
        optimizer. Additional hyperparameters can be passed in as keyword arguments, such as
        learning_rate, max_epochs, batch_size, and patience.

        Args:
            model: The PyTorch model to train.
            loss: The loss function to use.
            optimizer: The optimizer to use.
            resource_name: The name of the resource to create. Default is "model".
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(resource_name)
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = self.max_epochs = self.batch_size = self.patience = (
            self.monitor
        ) = None
        self._setup_hyperparameters(*args, **kwargs)

    # @hyperparameter("learning_rate", "max_epochs", "batch_size", "patience")
    def _setup_hyperparameters(
        self,
        learning_rate: float = 1e-4,
        max_epochs: int = 12,
        batch_size: int = 32,
        patience: int = 4,
        monitor: str = "valid_loss",
    ) -> None:
        """
        Set up the supervised training algorithm with the hyperparameters. These hyperparameters
        are used to configure the training process. The default values are provided, but can be
        overridden by the user.

        Args:
            learning_rate: The learning rate for the optimizer.
            max_epochs: The maximum number of epochs to train for (may terminate early).
            batch_size: The batch size to use.
            patience: The number of epochs to wait before early stopping.
            monitor: The metric to monitor for early stopping.

        Returns:
            None
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.monitor = monitor

    def __call__(
        self,
        model: torch.nn.Module,
        train_dataset: LabeledDataset,
        val_dataset: LabeledDataset,
        device: torch.device,
    ) -> NeuralNetRegressor:
        """
        Train the given model using the training dataset and validation dataset. The model is
        trained using the PyTorch model, loss function, optimizer, and hyperparameters provided
        during initialization.

        Args:
            model: The PyTorch model to train.
            train_dataset: Dataset for training.
            val_dataset: Dataset for validation.
            device: The device to use for training.

        Returns:
            model, losses [dictionary where keys are 'train' and 'val' for training losses (list),
            validation losses (list), respectively]
        """

        # create the skorch wrapper
        skorch_model = NeuralNetRegressor(
            model,
            criterion=self.loss,
            optimizer=self.optimizer,
            lr=self.learning_rate,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            train_split=predefined_split(val_dataset),
            device=device,
            callbacks=[EarlyStopping(patience=self.patience, monitor=self.monitor)],
        )

        skorch_model.fit(
            train_dataset.data.float().to(device=device),
            train_dataset.labels.float().to(device=device),
        )

        return skorch_model
