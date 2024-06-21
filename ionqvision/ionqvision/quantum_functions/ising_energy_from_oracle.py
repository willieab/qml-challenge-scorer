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

from abc import ABC, abstractmethod
from ionqvision.quantum_function_category import QuantumFunction
from ionqvision.utils import cvar_moments
from math import pi


def _attempt_bind(expr_or_float, val_dict):
    """
    Attempt to bind parameters in a Qiskit ParameterExpression.

    The attempt is needed because the output of the ParameterExpression.gradient
    method may be a float, which has no bind method.
    """
    if isinstance(expr_or_float, (float, np.floating)):
        return expr_or_float
    return expr_or_float.bind(val_dict, allow_unknown_parameters=True)


_vbind = np.vectorize(_attempt_bind)


class IsingEnergyFromOracle(QuantumFunction, ABC):
    r"""
    Base class for computing the expected energy of an Ising Hamiltonian 
    defined by its ``eigenvalues`` oracle with respect to the
    :class:`.VariationalAnsatz` ``ansatz``.

    Given an Ising Hamiltonian $H$ described by its :meth:`eigenvalues`
    $\lambda_x$, :meth:`fun` computes the expectation value 
    $$E(\theta) = \sum_x p_x \lambda_x,$$ 
    with $p_x = \vert\alpha_x(\theta)\vert^2$ denoting the probability of 
    observing the quantum state $\vert x \rangle$ when measuring the variational
    ``ansatz`` 
    $$
    \vert \psi(\theta)\rangle = \sum_x \alpha_x(\theta) \vert x \rangle.
    $$
    While the expectation is equivalent to 
    $$
    E(\theta) = \langle \psi(\theta) \vert \mathcal{H} \vert \psi(\theta) \rangle,
    $$
    the calculation is arranged so that it is not necessary to construct or
    even to know the full Hamiltonian $H$ in advance: all that is needed is
    an :meth:`eigenvalues` method that can lazily evaluate the appropriate 
    eigenvalues of $H$. 

    If ``ansatz.is_differentiable`` is ``True`` then :meth:`grad` is 
    available, and it implements the analytical gradient of the expected energy 
    with respect to the variational parameters via a parameter-shift rule.

    .. NOTE::

        This is an abstract class that requires every derived class to implement
        :meth:`eigenvalues` method. See, e.g., :class:`.IsingEnergy` or
        :class:`.ConstrainedIsingEnergy`.

    INPUT:

        - ``ansatz`` -- :class:`.VariationalAnsatz` determining the quantum state
    """
    def __init__(self, ansatz):
        super().__init__(ansatz)
        self.is_differentiable = self.ansatz.is_differentiable
            
    @abstractmethod
    def eigenvalues(self, states):
        """
        Compute the Hamiltonian eigenvalues corresponding to the given ``states``.

        Output must be a 1D-NumPy array with the same length as ``states``.
        """
        pass

    def from_measurements(measurements, cvar_alpha=1.0):
        """
        Compute the CVaR expectation value of the energy using a dictionary of
        measurements of ``self.ansatz``.
        """
        expec, _ = self._fun(measurements, cvar_alpha)
        return expec

    def _fun(self, measurements, cvar_alpha):
        """
        Evaluate the expected energy of ``self``'s Hamiltonian upon substituting 
        ``param_vals`` in place of the variational parameters.
        """
        assert len(measurements) == 1
        measurements = measurements[0]

        num_qubits = len(next(iter(measurements)))
        dtype = np.dtype([("states", int, (num_qubits,)), ("counts", "f")])
        res = np.fromiter(map(lambda kv: (list(kv[0]), kv[1]), measurements.items()), dtype)

        expec, var = cvar_moments(self.eigenvalues(res["states"]), res["counts"], cvar_alpha)
        return expec, var

    def _grad(self, measurements, cvar_alpha):
        """
        Evaluate the gradient of :meth:`_fun` by appropriately combining the
        given bound circuit ``measurements``
        """
        G = self.ansatz._param_map
        theta_vec = self.ansatz.get_param_order()
        m, n = G.shape[0], len(theta_vec)
        dG = np.zeros((m, n), dtype=object)
        for i, g in enumerate(G):
            for j, theta in enumerate(theta_vec):
                dG[i, j] = g.gradient(theta)

        vals = {theta: v for theta, v in zip(theta_vec, self._latest_param_vals)}
        dG = _vbind(dG, vals).astype(float)

        assert len(measurements) == 2*m
        dF, var = np.zeros(m), np.zeros(m)
        for k, kth_pair in enumerate(zip(measurements[0::2], measurements[1::2])):
            for j, pm_shift in enumerate(kth_pair):
                dFF, dFF_var = self._fun([pm_shift], cvar_alpha)
                dF[k] += (-1)**j * dFF
                var[k] += dFF_var

        grad = dF.T @ dG / 2
        grad_var = var.T @ dG**2
        assert len(grad) == len(grad_var)
        return grad, grad_var

    def _prepare_quantum_execution_fun(self, param_vals):
        """
        Get a list of all the bound circuits that must be executed to evaluate
        :meth:`_fun`.
        """
        return [self.ansatz.assign_parameters(param_vals)]

    def _prepare_quantum_execution_grad(self, param_vals):
        """
        Get a list of all the bound circuits that must be executed to evaluate
        :meth:`_fun`.
        """
        if isinstance(param_vals, torch.Tensor):
            param_vals = param_vals.detach().numpy()

        self._latest_param_vals = param_vals

        G, theta_vec = self.ansatz._param_map, self.ansatz.get_param_order()
        vals = {theta: v for theta, v in zip(theta_vec, param_vals)}
        G = _vbind(G, vals).astype(float)

        t_vec = self.ansatz._internal_params
        ansatz = self.ansatz._internal_ansatz
        bound_circ = list()
        for k in range(G.shape[0]):
            for pm in range(2):
                pm_shift = {t: g + (-1)**pm * (j == k)*pi/2 for j, (t, g) in enumerate(zip(t_vec, G))}
                bound_circ.append(ansatz.assign_parameters(pm_shift))
        return bound_circ

    def _fun_symb(self, param_vals=None):
        """
        Compute an analytical expression for ``self``'s Hamiltonian energy.
        """
        psi = self.ansatz.get_symbolic_expression()
        hamiltonian = sp.diag(self.eigenvalues().tolist(), unpack=True)
        return (psi.adjoint() * hamiltonian * psi)[0]
