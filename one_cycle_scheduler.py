from tensorflow.keras.optimizers import SGD, Adam
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import math


class OneCycleLr(keras.callbacks.Callback):
    def __init__(self, max_lr, total_steps, epochs=None, steps_per_epoch=None, pct_start=0.3, anneal_strategy="cos", cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=1e4):
        super(OneCycleLr, self).__init__()

        # Validate total steps:
        if total_steps is None and epochs is None and steps_per_epoch is None:
            raise ValueError("You must define either total_steps OR (epochs AND steps_per_epoch)")
        elif total_steps is not None:
            if total_steps <= 0 or not isinstance(total_steps, int):
                raise ValueError("Expected non-negative integer total_steps, but got {}".format(total_steps))
            self.total_steps = total_steps
        else:
            if epochs <= 0 or not isinstance(epochs, int):
                raise ValueError("Expected non-negative integer epochs, but got {}".format(epochs))
            if steps_per_epoch <= 0 or not isinstance(steps_per_epoch, int):
                raise ValueError("Expected non-negative integer steps_per_epoch, but got {}".format(steps_per_epoch))
            # Compute total steps
            self.total_steps = epochs * steps_per_epoch

        self.step_num = 0
        self.step_size_up = float(pct_start * self.total_steps) - 1
        self.step_size_down = float(self.total_steps - self.step_size_up) - 1

        # Validate pct_start
        if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
            raise ValueError("Expected float between 0 and 1 pct_start, but got {}".format(pct_start))

        # Validate anneal_strategy
        if anneal_strategy not in ["cos", "linear"]:
            raise ValueError("anneal_strategy must by one of 'cos' or 'linear', instead got {}".format(anneal_strategy))
        elif anneal_strategy == "cos":
            self.anneal_func = self._annealing_cos
        elif anneal_strategy == "linear":
            self.anneal_func = self._annealing_linear

        # Initialize learning rate variables
        self.initial_lr = max_lr / div_factor
        self.max_lr = max_lr
        self.min_lr = self.initial_lr / final_div_factor

        # Initial momentum variables
        self.cycle_momentum = cycle_momentum
        if self.cycle_momentum:
            self.m_momentum = max_momentum
            self.momentum = max_momentum
            self.b_momentum = base_momentum

    def _annealing_cos(self, start, end, pct) -> float:
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def _annealing_linear(self, start, end, pct) -> float:
        "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        return (end - start) * pct + start

    def set_lr_mom(self):
        if isinstance(self.model.optimizer, SGD):
            lr_attr = 'lr'
            mom_attr = 'momentum'
        elif isinstance(self.model.optimizer, Adam):
            lr_attr = 'learning_rate'
            mom_attr = 'beta_1'
        else:
            raise ValueError("Unsupported optimizer type")

        # Update learning rate
        if self.step_num <= self.step_size_up:
            computed_lr = self.anneal_func(
                self.initial_lr, self.max_lr, self.step_num / self.step_size_up
            )
        else:
            down_step_num = self.step_num - self.step_size_up
            computed_lr = self.anneal_func(
                self.max_lr, self.min_lr, down_step_num / self.step_size_down
            )
        K.set_value(self.model.optimizer.__dict__[lr_attr], computed_lr)

        # Update momentum if cycle_momentum is True
        if self.cycle_momentum:
            if self.step_num <= self.step_size_up:
                computed_momentum = self.anneal_func(
                    self.m_momentum, self.b_momentum, self.step_num / self.step_size_up
                )
            else:
                down_step_num = self.step_num - self.step_size_up
                computed_momentum = self.anneal_func(
                    self.b_momentum,
                    self.m_momentum,
                    down_step_num / self.step_size_down,
                )
            K.set_value(self.model.optimizer.__dict__[mom_attr], computed_momentum)
