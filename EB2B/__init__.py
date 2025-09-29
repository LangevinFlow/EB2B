"""Public interface for the EB2B Empirical Bayes package."""

from .empirical_bayes import EmpiricalBayesConfig, EmpiricalBayesKernelEstimator
from .trainer import EBTrainer, EBTrainingConfig
from .dip import skip

__all__ = [
    "EmpiricalBayesConfig",
    "EmpiricalBayesKernelEstimator",
    "EBTrainer",
    "EBTrainingConfig",
    "skip",
]
