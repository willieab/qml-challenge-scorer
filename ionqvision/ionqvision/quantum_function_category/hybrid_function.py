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
import torch

from qiskit.circuit import Parameter, ParameterVector
from qiskit.providers.backend import Backend as BuiltInQiskitBackend
from typing import List, Sequence, Tuple
from .quantum_function import QuantumFunction


class HybridFunction:
    """
    A kind of :class:`.QuantumFunction` that accepts two kinds of inputs: a data
    vector that is loaded onto the quantum computer using a sequence of
    parametric gates, and a vector with values for the trainable ansatz
    parameters.

    This class implements the bookkeeping needed in order to compute separate
    gradient vectors for each kind of input vector; that is, :meth:`grad`
    returns a pair of arrays, where the first one computes the gradient of the
    output with respect to the ``inputs`` array while the second one computes
    the gradient of the output with respect to the ``param_vals``, or trainable
    ansatz weights.

    This class is implemented mostly to serve as a bridge between
    :class:`.QuantumFunction`'s and the PyTorch development framework; it is not
    meant to be used directly but rather through :class:`.TorchQuantumFunction`.
    """
    def __init__(
            self,
            quantum_fun: QuantumFunction,
            input_params: List[List[Parameter]] | List[ParameterVector],
            trainable_weights: List[Parameter] | ParameterVector
        ):
        self._quantum_fun = quantum_fun

        self._weights = list(trainable_weights)
        self.set_input_param_order(input_params)

    def _flatten_vals(self, inputs, param_vals):
        assert all(len(x) == len(batch) for x, batch in zip(inputs, self.get_input_params())), "Mismatching number of input parameters given"
        assert len(param_vals) == len(self.get_trainable_weights()), "Mismatching number of trainable weights given"
        all_vals = (*inputs, param_vals)
        if isinstance(inputs[0], (list, tuple)):
            zero = type(inputs[0])()
            flat = sum(all_vals, zero)
        elif isinstance(inputs[0], torch.Tensor):
            flat = torch.cat(all_vals)
        elif isinstance(inputs[0], np.ndarray):
            flat = np.concatenate(all_vals)
        else:
            raise ValueError("inputs and param_vals must be specified as lists, tuples, NumPy arrays, or torch Tensors")
        return flat

    def _distinguish_grads(self, all_grads):
        accum = 0
        dinputs = list()
        for param_batch in self._input_params:
            dinputs.append(all_grads[accum:accum+len(param_batch)])
            accum += len(param_batch)
        dweights = all_grads[accum:]
        return *dinputs, dweights

    @property
    def ansatz(self):
        """
        Get the ansatz used to evaluate the quantum function.
        """
        return self._quantum_fun.ansatz

    def get_input_params(self):
        """
        Get a (ordered) list of lists corresponding to the input parameters.
        """
        return self._input_params

    def set_input_param_order(self, input_params):
        """
        Use a list of lists of ``input_params`` to specify the order in which
        input parameters should be considered for function evaluation and for
        computing partial derivatives to form the gradient vector.
        """
        self._input_params = list(map(list, input_params))

        flat_input_params = sum(self._input_params, list())
        self._quantum_fun.set_param_order(flat_input_params + self._weights)

    def get_trainable_weights(self):
        """
        Get a (ordered) list of trainable ansatz weights.
        """
        return self._weights

    def set_trainable_weight_order(self, weights):
        """
        Use a list of trainable ``weights`` to specify the order in which the
        trainable ansatz weights should be considered for function evaluation
        and when computing partial derivatives to form the gradient vector.
        """
        self._weights = list(weights)
        self.set_input_param_order(self._input_params)

    def fun(
            self,
            inputs: List[Sequence[float]],
            param_vals: Sequence[float],
            backend: BuiltInQiskitBackend | None = None,
            shots: int = 1000,
            cvar_alpha: float = 1.0,
        ) -> float:
        """
        Evaluate the hybrid function by substituting the ``inputs`` and ``param_vals``
        in the appropriate places.
        """
        vals = self._flatten_vals(inputs, param_vals)
        return self._quantum_fun.fun(vals, backend, shots, cvar_alpha)

    def fun_symb(
            self,
            inputs: List[Sequence[float]] | None = None,
            param_vals: Sequence[float] | None = None,
        ) -> sp.Expr:
        """
        Compute the symbolic form of this hybrid function.
        """
        vals = self._flatten_vals(inputs, param_vals) if inputs is not None else None
        return self._quantum_fun.fun_symb(vals)

    def grad(
            self,
            inputs: List[Sequence[float]],
            param_vals: Sequence[float],
            backend: BuiltInQiskitBackend | None = None,
            shots: int = 1000,
            cvar_alpha: float = 1.0,
        ) -> Tuple[List[Sequence[float]], Sequence[float]]:
        """
        Evaluate the hybrid function's gradient with respect to both ``inputs``
        and ``weights`` by substituting the ``inputs`` and ``param_vals`` in the
        appropriate places.
        """
        vals = self._flatten_vals(inputs, param_vals)
        all_grads = self._quantum_fun.grad(vals, backend, shots, cvar_alpha)
        assert len(all_grads) == sum(map(len, self._input_params)) + len(self._weights)
        return self._distinguish_grads(all_grads)

    def grad_symb(
            self,
            inputs: List[Sequence[float]] | None = None,
            param_vals: Sequence[float] | None = None,
        ) -> Tuple[List[sp.Matrix], sp.Matrix]:
        vals = self._flatten_vals(inputs, param_vals) if inputs is not None else None
        all_grads = self._quantum_fun.grad_symb(vals)
        assert len(all_grads) == sum(map(len, self._input_params)) + len(self._weights)
        return tuple(map(lambda g: sp.Matrix(g).T, self._distinguish_grads(all_grads)))
