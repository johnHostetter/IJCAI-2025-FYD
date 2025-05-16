"""
This file provides a wrapper to use neuro-fuzzy networks with the skorch library.
"""

from typing import Dict, Any

import torch
from skorch.history import History
from skorch import NeuralNetRegressor
from skorch.utils import _check_f_arguments, get_map_location


class SoftRegressor(NeuralNetRegressor):
    """
    A wrapper around skorch's NeuralNetRegressor that allows for the
    use of soft computing techniques, such as neuro-fuzzy systems and
    network morphisms.
    """

    def __init__(self, module, *args, criterion=torch.nn.MSELoss, **kwargs):
        super().__init__(module, *args, criterion=criterion, **kwargs)
        self.current_parameters: Dict[str, Any] = {}

    def module_parameters(self) -> Dict[str, Any]:
        """
        Get the parameters of the module.
        """
        return dict(self.module.named_parameters())

    def infer(self, x: torch.Tensor, **fit_params):
        """
        Infer the output of the module.
        """
        x = x.cuda() if torch.cuda.is_available() else x
        named_parameters = self.module_parameters()
        # generate any missing parameters needed for this batch
        self.module.eval()(x)
        if named_parameters.keys() != self.current_parameters.keys():
            # temporarily disable model initialization flag
            self.initialized_ = False
            # update optimizer with new parameters
            self._initialize_optimizer()
            # re-enable model initialization flag
            self.initialized_ = True
            # update current parameters
            self.current_parameters = named_parameters
        self.module.train()
        # proceed with inference
        return super().infer(x, **fit_params)

    def load_params(  # pylint: disable=too-many-arguments
        self,
        f_params=None,
        f_optimizer=None,
        f_criterion=None,
        f_history=None,
        checkpoint=None,
        use_safetensors=False,
        **kwargs,
    ):
        """Loads the module's parameters, history, and optimizer,
        not the whole object.

        To save and load the whole object, use pickle.

        ``f_params``, ``f_optimizer``, etc. uses PyTorch's
        :func:`~torch.load`.

        If you've created a custom module, e.g. ``net.mymodule_``, you
        can save that as well by passing ``f_mymodule``.

        Parameters
        ----------
        f_params : file-like object, str, None (default=None)
          Path of module parameters. Pass ``None`` to not load.

        f_optimizer : file-like object, str, None (default=None)
          Path of optimizer. Pass ``None`` to not load.

        f_criterion : file-like object, str, None (default=None)
          Path of criterion. Pass ``None`` to not save

        f_history : file-like object, str, None (default=None)
          Path to history. Pass ``None`` to not load.

        checkpoint : :class:`.Checkpoint`, None (default=None)
          Checkpoint to load params from. If a checkpoint and a ``f_*``
          path is passed in, the ``f_*`` will be loaded. Pass
          ``None`` to not load.

        use_safetensors : bool (default=False)
          Whether to use the ``safetensors`` library to load the state. By
          default, PyTorch is used, which in turn uses :mod:`pickle` under the
          hood. When the state was saved with ``safetensors=True`` when
          :meth:`skorch.net.NeuralNet.save_params` was called, it should be set
          to ``True`` here as well.

        Examples
        --------
        >>> before = NeuralNetClassifier(mymodule)
        >>> before.save_params(f_params='model.pkl',
        >>>                    f_optimizer='optimizer.pkl',
        >>>                    f_history='history.json')
        >>> after = NeuralNetClassifier(mymodule).initialize()
        >>> after.load_params(f_params='model.pkl',
        >>>                   f_optimizer='optimizer.pkl',
        >>>                   f_history='history.json')

        """
        if use_safetensors:
            raise NotImplementedError(
                "Loading with safetensors is not yet implemented in PySoft."
            )
        kwargs_full = self.__update_kwargs_with_checkpoint(checkpoint, f_history)

        # explicit arguments may override checkpoint arguments
        kwargs_full.update(**kwargs)
        for key, val in [
            ("f_params", f_params),
            ("f_optimizer", f_optimizer),
            ("f_criterion", f_criterion),
            ("f_history", f_history),
        ]:
            if val:
                kwargs_full[key] = val

        kwargs_module, kwargs_other = _check_f_arguments("load_params", **kwargs_full)

        if not kwargs_module and not kwargs_other:
            if self.verbose:
                print("Nothing to load")
            return

        # only valid key in kwargs_other is f_history
        f_history = kwargs_other.get("f_history")
        if f_history is not None:
            self.history = History.from_file(f_history)

        # this is the functionality that I have changed
        self.__load_from_common_state_dict(kwargs_module)

    def __update_kwargs_with_checkpoint(self, checkpoint, f_history):
        kwargs_full = {}
        if checkpoint is not None:
            if not self.initialized_:
                self.initialize()
            if f_history is None and checkpoint.f_history is not None:
                self.history = History.from_file(checkpoint.f_history_)
            kwargs_full.update(**checkpoint.get_formatted_files(self))
        return kwargs_full

    def __load_from_common_state_dict(self, kwargs_module) -> None:
        """
        Load the state dict from a file, but only load the parameters that exist in this module.
        """
        msg_init = (
            "Cannot load state of an un-initialized model. "
            "Please initialize first by calling .initialize() "
            "or by fitting the model with .fit(...)."
        )
        msg_module = (
            "You are trying to load 'f_{name}' but for that to work, the net "
            "needs to have an attribute called 'net.{name}_' that is a PyTorch "
            "Module or Optimizer; make sure that it exists and check for typos."
        )
        for attr, f_name in kwargs_module.items():
            # valid attrs can be 'module_', 'optimizer_', etc.
            if attr.endswith("_") and not self.initialized_:
                self.check_is_fitted([attr], msg=msg_init)
            module = self._get_module(attr, msg=msg_module)
            state_dict = self.__get_state_dict(f_name)
            try:
                module.load_state_dict(
                    {
                        key: value
                        for key, value in state_dict.items()
                        if key in module.state_dict()
                    }  # only keep what they share
                )
            except RuntimeError:
                pass  # we tried to load it but can't, let's continue

    def __get_state_dict(self, file):
        """
        Load the state dict from a file.
        """
        map_location = get_map_location(self.device)
        self.device = self._check_device(self.device, map_location)
        return torch.load(file, map_location=map_location)
