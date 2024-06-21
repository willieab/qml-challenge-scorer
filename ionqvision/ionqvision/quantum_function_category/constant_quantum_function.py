#
# IonQ, Inc., Copyright (c) 2024,
# All rights reserved.
# Use in source and binary forms of this software, without modification,
# is permitted solely for the purpose of activities associated with the IonQ
# Hackathon at Quantum Korea hosted by SKKU at Hotel Samjung and only during the
# June 21-23, 2024 duration of such event.
#
import numpy as np
import sympy as sp

from .quantum_function import QuantumFunction


class ConstantQuantumFunction(QuantumFunction):
    r"""
    A class to model the constant function $f(\theta) = c$, where $c$ is the
    (real) ``const``.
    """
    def __init__(self, ansatz, const):
        super().__init__(ansatz)
        self._const = const

    def _prepare_quantum_execution_fun(self, param_vals):
        return []

    def _fun(self, measurements, cvar_alpha):
        return self._const, 0

    def _prepare_quantum_execution_grad(self, param_vals):
        return []

    def _grad(self, measurements, cvar_alpha):
        nvars = self.ansatz.num_parameters
        return np.zeros(nvars), np.zeros(nvars)

    def _fun_symb(self, param_vals=None):
        return self._const
