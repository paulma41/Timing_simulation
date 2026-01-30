"""
Package providing function specifications and predefined functions.

Exports:
- Function: base class for function specifications
- build_h_function: factory for the h(t) function
"""

from .function_spec import Function
from .h_linear_exp import build_h_function
from .h_actions import build_h_action_function

__all__ = ["Function", "build_h_function", "build_h_action_function"]
