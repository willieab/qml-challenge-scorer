#
# IonQ, Inc., Copyright (c) 2024,
# All rights reserved.
# Use in source and binary forms of this software, without modification,
# is permitted solely for the purpose of activities associated with the IonQ
# Hackathon at Quantum Korea hosted by SKKU at Hotel Samjung and only during the
# June 21-23, 2024 duration of such event.
#
import numpy as np

from .constant_quantum_function import ConstantQuantumFunction
from .quantum_function import QuantumFunction


class QuantumFunctionProduct(QuantumFunction):
    """
    A class to model the sum of two :class:`.QuantumFunction` instances.
    """
    def __init__(self, lfactor, rfactor):
        assert lfactor.ansatz == lfactor.ansatz, "Factors must use the same ansatz"
        super().__init__(lfactor.ansatz)
        self._left = lfactor
        self._right = rfactor
        self._num_lqc_fun = None
        self._num_rqc_fun = None
        self._num_lqc_grad = None

    def _prepare_quantum_execution_fun(self, param_vals):
        left = self._left._prepare_quantum_execution_fun(param_vals)
        right = self._right._prepare_quantum_execution_fun(param_vals)
        self._num_lqc_fun = len(left)
        self._num_rqc_fun = len(right)
        return left + right

    def _fun(self, measurements, cvar_alpha):
        lfun, lvar = self._left._fun(measurements[:self._num_lqc_fun], cvar_alpha)
        rfun, rvar = self._right._fun(measurements[self._num_lqc_fun:], cvar_alpha)

        if isinstance(self._left, ConstantQuantumFunction):
            var = self._left._const**2 * rvar
        elif isinstance(self._right, ConstantQuantumFunction):
            var = self._right._const**2 * lvar
        else:
            var = np.nan
        return lfun * rfun, var

    def _prepare_quantum_execution_grad(self, param_vals):
        left_fun = self._left._prepare_quantum_execution_fun(param_vals)
        right_fun = self._right._prepare_quantum_execution_fun(param_vals)
        left = self._left._prepare_quantum_execution_grad(param_vals)
        right = self._right._prepare_quantum_execution_grad(param_vals)
        self._num_lqc_grad = len(left)
        return left_fun + right_fun + left + right

    def _grad(self, measurements, cvar_alpha):
        lqc, rqc = self._num_lqc_fun, self._num_rqc_fun
        lfun, _ = self._left._fun(measurements[:lqc], cvar_alpha)
        rfun, _ = self._right._fun(measurements[lqc:lqc+rqc], cvar_alpha)

        lgrad, lvar = self._left._grad(measurements[lqc+rqc:lqc+rqc+self._num_lqc_grad], cvar_alpha)
        rgrad, rvar = self._right._grad(measurements[lqc+rqc+self._num_lqc_grad:], cvar_alpha)

        if isinstance(self._left, ConstantQuantumFunction):
            var = self._left._const**2 * rvar
        elif isinstance(self._right, ConstantQuantumFunction):
            var = self._right._const**2 * lvar
        else:
            var = np.array([np.nan]*len(lgrad))

        return lgrad * rfun + rgrad * lfun, var

    def _fun_symb(self, param_vals=None):
        left = self._left._fun_symb(param_vals)
        right = self._right._fun_symb(param_vals)
        return left * right
