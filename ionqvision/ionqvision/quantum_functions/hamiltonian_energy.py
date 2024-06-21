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

from collections import defaultdict
from ionqvision.utils import change_of_basis_circuit
from .ising_energy import IsingEnergy

from qiskit_aer import Aer
from ionqvision.quantum_function_category import QuantumFunction


class HamiltonianEnergy(QuantumFunction):
    r"""
    Module for computing the expected energy of a ``hamiltonian`` with respect
    to a variational ``ansatz``.

    Precisely, :meth:`fun` evaluates the expected value
    $$
    E(\theta) = \langle \psi(\theta) \vert H \vert \psi(\theta) \rangle,
    $$
    with $H$ denoting the ``hamiltonian`` and $\vert \psi(\theta) \rangle$
    denoting the ``ansatz``.

    The ``hamiltonian`` may be specified in two ways. If ``hamiltonian`` is a
    Qiskit :class:`SparsePauliOp` the Hamiltonian is split into qubit-wise
    commuting term groups and this module imputes an appropriate Pauli
    measurement basis for each term group.

    Alternatively, the ``hamiltonian`` may be specified as a list of tuples
    ``(comm_term, measurement_qc)`` where each ``comm_term`` is a
    :class:`SparsePauliOp` describing a group of commuting terms and
    ``measurement_qc`` is a :class:`QuantumCircuit` implementing a unitary
    transformation that diagonalizes the corresponding ``comm_term``.

    The ``ansatz`` should be a 
    :class:`.VariationalAnsatz` instance, as these
    objects keep track of the information needed to compute the gradient of the
    expected energy with respect to the variational parameters.
    """
    def __init__(self, hamiltonian, ansatz):
        super().__init__(ansatz)
        self._hamiltonian = hamiltonian
        self.is_differentiable = ansatz.is_differentiable

        if isinstance(hamiltonian, list):
            self._comm_gp, self._measurement_bases = list(zip(*hamiltonian))
        else:
            self._comm_gp = hamiltonian.group_commuting(qubit_wise=True)
            self._measurement_bases = list()
            for qwc_gp in self._comm_gp:
                basis = ""
                words = np.array([list(word) for word in qwc_gp.paulis.to_labels()])
                for qubit in range(words.shape[1]):
                    qubit_ops = "".join(filter(lambda sig: sig != "I", words[:, qubit]))
                    basis += qubit_ops[0] if qubit_ops else "Z"
                self._measurement_bases.append(basis)

        self._gp_energies = list()
        for gp, basis in zip(self._comm_gp, self._measurement_bases):
            if isinstance(basis, str):
                basis = change_of_basis_circuit(basis)
            gp_ansatz = ansatz.compose(basis)
            self._gp_energies.append(IsingEnergy(gp, gp_ansatz))

        self._gp_metadata = [defaultdict(list) for _ in range(len(self._gp_energies))]

    def _prepare_quantum_execution_fun(self, param_vals):
        """
        Get the list of bound ``QuantumCircuit``'s we need to execute to
        evaluate the energy.
        """
        return [e._prepare_quantum_execution_fun(param_vals)[0] for e in self._gp_energies]

    def _fun(self, measurements, cvar_alpha):
        """
        Obtain the energy as a function of the list of ``measurements`` of the
        circuits obtained using :meth:`_prepare_quantum_execution_fun`.
        """
        assert len(measurements) == len(self._gp_energies)
        energy, var = 0., 0.
        for j, (ising_like, qc_meas) in enumerate(zip(self._gp_energies, measurements)):
            gp_energy, gp_var = ising_like._fun([qc_meas], cvar_alpha)
            energy += gp_energy

            if np.isclose(gp_var, 0, atol=1e-5):
                gp_var = 0.0
            assert gp_var >= 0, gp_var
            var += gp_var

            shots = round(sum(qc_meas.values()))
            for attr, val in zip(["energy", "var", "shots"], [gp_energy, gp_var, shots]):
                self._gp_metadata[j][attr].append(val)
            if shots == 1:
                gp_var = 0
            self._gp_metadata[j]["std_err"].append(np.sqrt(gp_var / shots))
        return energy, var

    def _prepare_quantum_execution_grad(self, param_vals):
        """
        Get the list of quantum circuits that need to be executed to evaluate
        the gradient of the energy w.r.t. the variational parameters-- this
        is just the sum of the circuits required by each Ising-like term.
        """
        self._latest_param_vals = param_vals
        return sum([e._prepare_quantum_execution_grad(param_vals) for e in self._gp_energies], list())

    def _grad(self, measurements, cvar_alpha):
        """
        Evaluate the gradient of the energy w.r.t. the variational parameters
        given the measurements of all the circuits returned by
        :meth:`_prepare_quantum_execution_grad`
        """
        assert len(measurements) % len(self._gp_energies) == 0

        grad, var = 0., 0.
        per_term = len(measurements) // len(self._gp_energies)
        for j, ising_like in enumerate(self._gp_energies):
            gp_grad, gp_var = ising_like._grad(measurements[j*per_term:(j+1)*per_term], cvar_alpha)
            grad += gp_grad
            var += gp_var
        return grad, var

    def _fun_symb(self, param_vals=None):
        """
        Evaluate the symbolic form of :meth:`fun`. 

        When ``param_values`` is ``None``, method returns a symbolic
        expression for the Hamiltonian energy.
        """
        return sum(energy._fun_symb(param_vals) for energy in self._gp_energies)

    @property
    def commuting_term_groups(self):
        """
        Get a list of ``(commuting_term_group, measurement_basis)`` describing
        ``self.hamiltonian``.
        """
        return list(zip(self._comm_gp, self._measurement_bases))

    @property
    def group_metadata(self):
        """
        Get a list of metadata dictionaries corresponding to each Ising-like
        term in :meth:`ising_like_energies`.
        """
        return self._gp_metadata

    @property
    def hamiltonian(self):
        """
        Get the Hamiltonian whose energy ``self`` evaluates.
        """
        return self._hamiltonian

    @property
    def ising_like_energies(self):
        """
        Get the list of :class:`.IsingEnergy` objects used to evaluate the
        energy of ``self.hamiltonian``.
        """
        return self._gp_energies

    def set_param_order(self, param_list):
        """
        Set the order in which parameter lists should be interpreted.
        """
        self.ansatz.set_param_order(param_list)
        [energy.set_param_order(param_list) for energy in self._gp_energies]
