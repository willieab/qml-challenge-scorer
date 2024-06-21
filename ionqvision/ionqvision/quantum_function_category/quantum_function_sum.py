#
# IonQ, Inc., Copyright (c) 2024,
# All rights reserved.
# Use in source and binary forms of this software, without modification,
# is permitted solely for the purpose of activities associated with the IonQ
# Hackathon at Quantum Korea hosted by SKKU at Hotel Samjung and only during the
# June 21-23, 2024 duration of such event.
#
from .quantum_function import QuantumFunction


class QuantumFunctionSum(QuantumFunction):
    """
    A class to model the sum of two :class:`.QuantumFunction` instances.
    """
    def __init__(self, lsummand, rsummand):
        assert lsummand.ansatz == rsummand.ansatz, "Summands must use the same ansatz"
        super().__init__(lsummand.ansatz)
        self._left = lsummand
        self._right = rsummand
        self._num_lqc_fun = None
        self._num_lqc_grad = None

    def _prepare_quantum_execution_fun(self, param_vals):
        left = self._left._prepare_quantum_execution_fun(param_vals)
        right = self._right._prepare_quantum_execution_fun(param_vals)
        self._num_lqc_fun = len(left)
        return left + right

    def _fun(self, measurements, cvar_alpha):
        lfun, lvar = self._left._fun(measurements[:self._num_lqc_fun], cvar_alpha)
        rfun, rvar = self._right._fun(measurements[self._num_lqc_fun:], cvar_alpha)
        return lfun + rfun, lvar + rvar

    def _prepare_quantum_execution_grad(self, param_vals):
        left = self._left._prepare_quantum_execution_grad(param_vals)
        right = self._right._prepare_quantum_execution_grad(param_vals)
        self._num_lqc_grad = len(left)
        return left + right

    def _grad(self, measurements, cvar_alpha):
        lgrad, lvar = self._left._grad(measurements[:self._num_lqc_grad], cvar_alpha)
        rgrad, rvar = self._right._grad(measurements[self._num_lqc_grad:], cvar_alpha)
        assert len(lgrad) == len(lvar) == len(rgrad) == len(rvar), (lgrad, lvar, rgrad, rvar)
        return lgrad + rgrad, lvar + rvar

    def _fun_symb(self, param_vals=None):
        left = self._left._fun_symb(param_vals)
        right = self._right._fun_symb(param_vals)
        return left + right
