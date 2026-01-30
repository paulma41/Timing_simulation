"""
Design optimizers for experimental designs.

Exports:
- BaseDesignOptimizer: abstract base interface
- FisherOptimizer: optimizer maximizing Fisher information criterion
- fisher_optimizer: convenience function wrapper
- Fisher_info_optimizer: alias for backward-compatibility
"""

from .base import BaseDesignOptimizer
from .fisher import FisherOptimizer, fisher_optimizer, Fisher_info_optimizer
from .jeffreys import (
    expected_log_bayes_factor_for_design,
    jeffreys_optimizer,
    expected_log_bayes_factor_matrix_for_design,
    jeffreys_optimizer_multi,
    jeffreys_optimizer_multi_bo,
)
from .laplace_jsd import laplace_jsd_for_design, laplace_jsd_matrix_for_design, laplace_jsd_separability

__all__ = [
    "BaseDesignOptimizer",
    "FisherOptimizer",
    "fisher_optimizer",
    "Fisher_info_optimizer",
    "expected_log_bayes_factor_for_design",
    "jeffreys_optimizer",
    "expected_log_bayes_factor_matrix_for_design",
    "jeffreys_optimizer_multi",
    "jeffreys_optimizer_multi_bo",
    "laplace_jsd_for_design",
    "laplace_jsd_matrix_for_design",
    "laplace_jsd_separability",
]
