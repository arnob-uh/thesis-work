"""
https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
"""
import numpy as np
import torch

class EarlyStopping:
    def __init__(
        self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def get_state(self):
        return {
            "counter": self.counter,
            "best_score": self.best_score,
            "val_loss_min": self.val_loss_min,
            "early_stop": self.early_stop
        }

    def set_state(self, state):
        self.counter = state["counter"]
        self.best_score = state["best_score"]
        self.val_loss_min = state["val_loss_min"]
        self.early_stop = state["early_stop"]

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
