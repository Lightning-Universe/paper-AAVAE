from typing import Any, Dict, List

import numpy as np
import torch

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class EarlyStopping(Callback):
    mode_dict = {
        'min': torch.lt,
        'max': torch.gt,
    }

    def __init__(
        self,
        monitor1: str = 'early_stop_on',
        monitor2: str = 'early_stop_on',
        min_delta1: float = 0.0,
        min_delta2: float = 0.0,
        patience: int = 3,
        verbose: bool = False,
        mode1: str = 'min',
        mode2: str = 'min',
        strict: bool = True,
    ):
        super().__init__()
        self.monitor1 = monitor1
        self.monitor2 = monitor2
        self.patience = patience
        self.verbose = verbose
        self.strict = strict
        self.min_delta1 = min_delta1
        self.min_delta2 = min_delta2
        self.wait_count1 = 0
        self.wait_count2 = 0
        self.stopped_epoch = 0
        self.mode1 = mode1
        self.mode2 = mode2

        if (self.mode1 not in self.mode_dict) or (self.mode2 not in self.mode_dict):
            raise MisconfigurationException(f"`mode` can be {', '.join(self.mode_dict.keys())}")

        torch_inf = torch.tensor(np.Inf)

        self.min_delta1 *= 1 if self.monitor_op1 == torch.gt else -1
        self.min_delta2 *= 1 if self.monitor_op2 == torch.gt else -1

        self.best_score1 = torch_inf if self.monitor_op1 == torch.lt else -torch_inf
        self.best_score2 = torch_inf if self.monitor_op2 == torch.lt else -torch_inf

    def _validate_condition_metric(self, logs):
        monitor_val1 = logs.get(self.monitor1)
        monitor_val2 = logs.get(self.monitor2)

        error_msg = (
            f'Early stopping conditioned on metric `{self.monitor1}` and `{self.monitor2}` which is not available.'
            ' Pass in or modify your `EarlyStopping` callback to use any of the following:'
            f' `{"`, `".join(list(logs.keys()))}`'
        )

        if monitor_val1 is None or monitor_val2 is None:
            if self.strict:
                raise RuntimeError(error_msg)
            if self.verbose > 0:
                rank_zero_warn(error_msg, RuntimeWarning)

            return False

        return True

    @property
    def monitor_op1(self):
        return self.mode_dict[self.mode1]

    @property
    def monitor_op2(self):
        return self.mode_dict[self.mode2]

    def on_save_checkpoint(self, trainer, pl_module, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'wait_count1': self.wait_count1,
            'wait_count2': self.wait_count2,
            'stopped_epoch': self.stopped_epoch,
            'best_score1': self.best_score1,
            'best_score2': self.best_score2,
            'patience': self.patience
        }

    def on_load_checkpoint(self, callback_state: Dict[str, Any]):
        self.wait_count1 = callback_state['wait_count1']
        self.wait_count2 = callback_state['wait_count2']
        self.stopped_epoch = callback_state['stopped_epoch']
        self.best_score1 = callback_state['best_score1']
        self.best_score2 = callback_state['best_score2']
        self.patience = callback_state['patience']

    def on_validation_end(self, trainer, pl_module):
        from pytorch_lightning.trainer.states import TrainerState
        if trainer.state != TrainerState.FITTING or trainer.sanity_checking:
            return

        self._run_early_stopping_check(trainer)

    def _run_early_stopping_check(self, trainer):
        """
        Checks whether the early stopping condition is met
        and if so tells the trainer to stop the training.
        """
        logs = trainer.callback_metrics

        if (
            trainer.fast_dev_run  # disable early_stopping with fast_dev_run
            or not self._validate_condition_metric(logs)  # short circuit if metric not present
        ):
            return  # short circuit if metric not present

        current1 = logs.get(self.monitor1)
        current2 = logs.get(self.monitor2)

        # when in dev debugging
        # comment out for multiple metrics for now
        # trainer.dev_debugger.track_early_stopping_history(self, current)

        if self.monitor_op1(current1 - self.min_delta1, self.best_score1):
            self.best_score1 = current1
            self.wait_count1 = 0
        else:
            self.wait_count1 += 1

        if self.monitor_op2(current2 - self.min_delta2, self.best_score2):
            self.best_score2 = current2
            self.wait_count2 = 0
        else:
            self.wait_count2 += 1

        # "and" logic for 2 metrics
        if self.wait_count1 >= self.patience and self.wait_count2 >= self.patience:
            self.stopped_epoch = trainer.current_epoch
            trainer.should_stop = True

        # stop every ddp process if any world process decides to stop
        trainer.should_stop = trainer.training_type_plugin.reduce_boolean_decision(trainer.should_stop)
