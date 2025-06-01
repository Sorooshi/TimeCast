import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple


class BaseTimeSeriesModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.model_name = self.__class__.__name__

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        pass

    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """Return default hyperparameters for the model."""
        pass

    @abstractmethod
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer for the model."""
        pass

    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Return parameter ranges for Optuna hyperparameter tuning."""
        pass 