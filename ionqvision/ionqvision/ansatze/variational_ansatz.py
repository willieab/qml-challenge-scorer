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

from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from ionqvision.utils import change_of_basis_circuit, robust_backend_call
from qiskit import QuantumCircuit, qpy
from qiskit_aer import Aer
from qiskit.circuit import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.compiler import transpile
from qiskit.quantum_info import Operator
from typing import Sequence


sig = {
    "I": sp.Matrix([[1, 0], [0, 1]]),
    "X": sp.Matrix([[0, 1], [1, 0]]),
    "Y": sp.Matrix([[0, -sp.I], [sp.I, 0]]),
    "Z": sp.Matrix([[1, 0], [0, -1]])
}
pauli = lambda word: sp.kronecker_product(*[sig[ax] for ax in word])
rot_mat = lambda ax, var: pauli('I'*len(ax))*sp.cos(var/2) - sp.I*sp.sin(var/2)*pauli(ax)


COMPOSITE_GATES = ["CONV", "Givens", "POOL", "RBS", "TSP"]

COMPOSITE_GATES += ["crx", "cry", "crz", "crx_o0", "cry_o0", "crz_o0"]

COMPOSITE_GATES += [gate + "_dg" for gate in COMPOSITE_GATES]


class VariationalAnsatz(QuantumCircuit):
    r"""
    A class to manage variational quantum ansatzes.

    Suppose your ansatz takes the special form

    .. math::
        :name: diffn_ansatz

        \vert \psi(\theta) \rangle = V_m(g_m(\theta)) S_m 
        \cdots V_1(g_1(\theta)) S_1 \vert 0 \rangle.
    
    Here each variational component $V_j$ is given by
    $$
    V_j(s) = e^{-\frac{i}{2} s P_j},
    $$
    where $P_j = \sum_k c_{jk} \omega^{jk}$ is a (real) linear combination of
    **commuting** Pauli words $\omega^{jk}$, and each static component $S_j$ is
    an arbitrary non-variational unitary trasnformation. In addition, each
    $g_j: \mathbb{R}^n \to \mathbb{R}$ denotes an arbitrary smooth map. The
    parameter map $G(\theta) = 
    \begin{bmatrix} g_1(\theta) & \cdots & g_m(\theta) \end{bmatrix}^T$ makes it
    so that we may correlate the angles supplied to the parametric gates in the
    variational circuit. For an example, see :meth:`is_differentiable`.

    Then this module keeps track of the necessary information in order to
    differentiate an expected energy computation using a parameter-shift rule.
    
    Examples include QAOA ansatze and their
    multi-angle relatives. The :class:`.HamiltonianEnergy` class computes the
    gradient of the expected energy of such variational states using a
    parameter-shift rule: each partial derivative is a linear combination
    (summing across gates in the same layer) of energy function evaluations
    at different points.

    :class:`VariationalAnsatz` inherits its constructor from ``QuantumCircuit``.

    :EXAMPLE:
        
        >>> from ionqvision.ansatze import VariationalAnsatz
        >>> qc = VariationalAnsatz(2, 1)
        >>> _ = qc.x(1)
        >>> qc.draw()
        <BLANKLINE>
        q_0: ─────
             ┌───┐
        q_1: ┤ X ├
             └───┘
        c: 1/═════
        <BLANKLINE>
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._param_list = None
        self._param_symbols = None

    def _construct_parameter_map(self):
        r"""
        Construct the parameter map $\theta \to G(\theta)$ determining the
        angles input to the parameterized gates in ``self._internal_ansatz``.

        This method is meant for internal use only.
        """
        self._internal_ansatz = QuantumCircuit(self.num_qubits)
        self._internal_params = list()
        param_map = list()
        
        BASIS = ["rx", "ry", "rz", "cx", "rxx", "ryy", "rzz"]
        basis_qc = transpile(self, basis_gates=BASIS, optimization_level=3)

        for instruction in basis_qc:
            num_free_params = sum(isinstance(p, ParameterExpression) for p in instruction.operation.params)

            if num_free_params == 0:
                self._internal_ansatz.append(instruction)
            elif num_free_params == 1 and instruction.operation.name[0] == "r":
                param_map.append(instruction.operation.params[0])

                t = Parameter("t_" + str(len(self._internal_params)))
                internal_instr = deepcopy(instruction)
                internal_instr.operation.params[0] = t
                self._internal_ansatz.append(internal_instr)
                self._internal_params.append(t)
            else:
                self._is_differentiable = False
                return

        self._is_differentiable = len(param_map) > 0
        self._param_map = np.array(param_map)

    def assign_parameters(self, param_vals):
        """
        Assign the ``param_vals`` to the variational parameters in ``self``.

        When ``param_vals`` is a ``list`` or an ``np.ndarray``, the values are
        assumed to be sorted according to :meth:`get_param_order`. As usual,
        ``param_vals`` can also be a dictionary of parameter-value pairs.

        .. NOTE::

            This method overloads the ``QuantumCircuit.assign_parameters``
            method so that ``param_vals`` may be arranged according to the
            preferred order fixed by :meth:`set_param_order`.

        :EXAMPLE:

            >>> from ionqvision.ansatze import VariationalAnsatz
            >>> from qiskit.circuit import ParameterVector
            >>> t = ParameterVector("t", 2)
            >>> ansatz = VariationalAnsatz(1)
            >>> _ = ansatz.rx(t[0], 0)
            >>> _ = ansatz.ry(t[1], 0)
            >>> ansatz.draw()
               ┌──────────┐┌──────────┐
            q: ┤ Rx(t[0]) ├┤ Ry(t[1]) ├
               └──────────┘└──────────┘
            >>> ansatz.get_param_order()
            [ParameterVectorElement(t[0]), ParameterVectorElement(t[1])]
            >>> ansatz.assign_parameters([0, 1]).draw()
               ┌───────┐┌───────┐
            q: ┤ Rx(0) ├┤ Ry(1) ├
               └───────┘└───────┘
            >>> ansatz.set_param_order([t[1], t[0]])
            >>> ansatz.get_param_order()
            [ParameterVectorElement(t[1]), ParameterVectorElement(t[0])]
            >>> ansatz.assign_parameters([0, 1]).draw()
               ┌───────┐┌───────┐
            q: ┤ Rx(1) ├┤ Ry(0) ├
               └───────┘└───────┘
            >>> ansatz.assign_parameters({t[0]: 1, t[1]: 0}).draw()
               ┌───────┐┌───────┐
            q: ┤ Rx(1) ├┤ Ry(0) ├
               └───────┘└───────┘
        """
        if isinstance(param_vals, (Sequence, np.ndarray, torch.Tensor)):
            param_vals = {p: float(v) for p, v in zip(self.get_param_order(), param_vals)}
        return super().assign_parameters(param_vals)

    @classmethod
    def from_file(self, path):
        with open(path, "rb") as f:
            ansatz = qpy.load(f)
        return ansatz[0]

    @classmethod
    def from_quantum_circuit(cls, qc):
        """
        Construct a :class:`VariationalAnsatz` instance identical to the
        ``QuantumCircuit`` ``qc``.

        :EXAMPLE:

            >>> from qiskit import QuantumCircuit
            >>> from qiskit.circuit import Parameter
            >>> qc = QuantumCircuit(2)
            >>> _ = qc.cnot(1, 0)
            >>> theta = Parameter("t")
            >>> _ = qc.rzz(theta, 0, 1)
            >>> qc.draw()
                 ┌───┐        
            q_0: ┤ X ├─■──────
                 └─┬─┘ │ZZ(t) 
            q_1: ──■───■──────
            <BLANKLINE>
            >>> from ionqvision.ansatze import VariationalAnsatz
            >>> ansatz = VariationalAnsatz.from_quantum_circuit(qc)
            >>> ansatz.draw()
                 ┌───┐        
            q_0: ┤ X ├─■──────
                 └─┬─┘ │ZZ(t) 
            q_1: ──■───■──────
            <BLANKLINE>
            >>> list(qc) == list(ansatz)
            True
        """
        ansatz = VariationalAnsatz(qc.num_qubits)
        for instruction in qc:
            ansatz.append(instruction)
        return ansatz

    def get_param_order(self):
        """
        Get an (ordered) list of the parameters in ``self``.

        .. NOTE::

            This method defines the evaluation order of the variational
            parameters; see :meth:`assign_parameters` for details.
        """
        if self._param_list is None:
            return list(self.parameters)
        if not len(self._param_list) == len(self.parameters):
            raise ValueError("The number of parameters has changed. Re-set their order.")
        return self._param_list

    def set_param_order(self, param_list):
        """
        Fix the parameter evaluation order according to ``param_list``.

        Here ``param_list`` must be a list or a ``ParameterVector`` such that
        ``set(param_list) == set(self.parameters)``.

        .. NOTE::

            This method changes the evaluation order of the variational
            parameters; see :meth:`assign_parameters` for details.
        """
        if not set(param_list) == set(self.parameters):
            raise ValueError(f"Invalid param_list: it must be a permutation of self.parameters")
        self._param_list = list(param_list)

    def get_symbolic_expression(self):
        """
        Compute a symbolic SymPy ``2**self.num_qubits``-dimensional state vector
        describing ``self``.

        :EXAMPLE:
            
            >>> from ionqvision.ansatze import VariationalAnsatz
            >>> from qiskit.circuit import Parameter
            >>> theta = Parameter("t")
            >>> ansatz = VariationalAnsatz(2)
            >>> _ = ansatz.h(0)
            >>> _ = ansatz.ryy(theta, 0, 1)
            >>> _ = ansatz.cnot(0, 1)
            >>> ansatz.draw()
                 ┌───┐┌─────────┐     
            q_0: ┤ H ├┤0        ├──■──
                 └───┘│  Ryy(t) │┌─┴─┐
            q_1: ─────┤1        ├┤ X ├
                      └─────────┘└───┘
            >>> ansatz.get_symbolic_expression()
            Matrix([
            [   0.707106781186547*cos(t/2)],
            [ 0.707106781186547*I*sin(t/2)],
            [-0.707106781186547*I*sin(t/2)],
            [   0.707106781186547*cos(t/2)]])
        """
        psi = sp.Matrix([1 if j == 0 else 0 for j in range(2**self.num_qubits)])
        qc_params = list()
        self._param_symbols = {}
        for instruction in self.decompose(COMPOSITE_GATES, reps=2):
            if instruction.operation.name == "barrier":
                continue

            num_free_params = sum(isinstance(p, ParameterExpression) for p in instruction.operation.params)
            if num_free_params == 0:
                op = QuantumCircuit(self.num_qubits)
                op.append(instruction)
                unitary = sp.Matrix(Operator(op).data)
                
            elif num_free_params == 1 and instruction.operation.name[0] == "r":
                word = list("I" * self.num_qubits)
                for ax, qubit in zip(instruction.operation.name[1:], instruction.qubits):
                    idx = self.find_bit(qubit).index
                    word[idx] = ax.upper()
                word = "".join(reversed(word))

                gate_param = instruction.operation.params[0]
                self._param_symbols |= gate_param._parameter_symbols
                unitary = rot_mat(word, gate_param.sympify())
            else:
                raise NotImplementedError(f"Only single parameter multi-qubit rotations are supported; {instruction.operation.name} is not.")

            psi = unitary * psi

        for psi_symb in psi.free_symbols:
            for k, v in self._param_symbols.items():
                if psi_symb == v:
                    self._param_symbols[k] = psi_symb
            
        real_symbols = {t: sp.symbols(t.name, real=True) for t in psi.free_symbols}
        self._param_symbols = {p: real_symbols[self._param_symbols[p]] for p in self.parameters}
        psi = psi.subs(real_symbols)
        return psi

    def givens(self, theta, qubit1, qubit2):
        r"""
        Apply a Givens rotation gate with the angle ``theta`` onto ``qubit1``
        and ``qubit2``.

        This gate acts as a counter-clockwise rotation by an angle of ``theta``
        in the (real) $(\vert 01 \rangle, \vert 10 \rangle)$-plane of the
        subspace defined by the tensor product of the state spaces corresponding
        to ``qubit1`` and ``qubit2``. In other words, the following matrix
        describes the transformation defined by ``givens(theta, 0, 1)``:

        .. math::

            \begin{bmatrix}
            1 & 0 & 0 & 1 \\
            0 & \cos(\theta) & -\sin(\theta) & 0 \\
            0 & \sin(\theta) & \cos(\theta) & 0 \\
            1 & 0 & 0 & 1
            \end{bmatrix}.

        :EXAMPLE:

            >>> from ionqvision.ansatze import VariationalAnsatz
            >>> from qiskit.circuit import Parameter
            >>> ansatz = VariationalAnsatz(2)
            >>> _ = ansatz.x(0)
            >>> _ = ansatz.givens(Parameter("t"), 0, 1)
            >>> ansatz.draw()
                 ┌───┐┌────────────┐
            q_0: ┤ X ├┤0           ├
                 └───┘│  Givens(t) │
            q_1: ─────┤1           ├
                      └────────────┘
            >>> expr = ansatz.get_symbolic_expression()
            >>> expr.simplify(); expr
            Matrix([
            [             0],
            [1.0*cos(1.0*t)],
            [1.0*sin(1.0*t)],
            [             0]])
        """
        sub_circ = QuantumCircuit(2, name="Givens")
        sub_circ.s(0)
        sub_circ.s(1)
        sub_circ.h(1)
        sub_circ.cx(1, 0)
        sub_circ.ry(-theta, 0)
        sub_circ.ry(-theta, 1)
        sub_circ.cx(1, 0)
        sub_circ.h(1)
        sub_circ.sdg(1)
        sub_circ.sdg(0)
        return self.append(sub_circ.to_instruction(), [qubit1, qubit2])

    @property
    def is_differentiable(self):
        """
        Determine whether ``self`` is differentiable.

        Any ansatz of the form :ref:`Equation (1) <diffn_ansatz>` is
        differentiable. As mentioned above, the parameter map $G$ allows for
        differentiating circuits with correlated parameters.

        :EXAMPLE:

            >>> from ionqvision.ansatze import VariationalAnsatz
            >>> from qiskit.circuit import ParameterVector
            >>> theta = ParameterVector("t", 2)
            >>> ansatz = VariationalAnsatz(3)
            >>> _ = ansatz.h(0)
            >>> _ = ansatz.y(2)
            >>> _ = ansatz.rzz(theta[0] - 3*theta[1], 0, 2)
            >>> _ = ansatz.cnot(1, 2)
            >>> _ = ansatz.rxx(theta[1]*theta[0], 0, 1)
            >>> ansatz.draw()
                 ┌───┐                         ┌─────────────────┐
            q_0: ┤ H ├─■───────────────────────┤0                ├
                 └───┘ │                       │  Rxx(t[0]*t[1]) │
            q_1: ──────┼────────────────────■──┤1                ├
                 ┌───┐ │ZZ(t[0] - 3*t[1]) ┌─┴─┐└─────────────────┘
            q_2: ┤ Y ├─■──────────────────┤ X ├───────────────────
                 └───┘                    └───┘                   
            >>> ansatz.is_differentiable
            True
        """
        self._construct_parameter_map()
        return self._is_differentiable
    
    def measure_ansatz(self, param_vals, basis=None, backend=None, shots=1000, record_id="", internal=False):
        """
        Bind parameters and measure variational circuit.

        INPUT:

            - ``param_vals`` -- list or dictionary of variational parameter
              values
            - ``backend`` -- (optional) Qiskit backend target for executing
              ansatz; defaults to Qiskit's ``aer_simulator``
            - ``shots`` -- (optional) number of quantum circuit measurements
            - ``basis`` -- (optional) string indicating a Pauli measurement
              basis for each qubit or a ``QuantumCircuit`` implementing the
              appropriate basis change operation; by default, all qubits are
              measured in the Pauli-Z basis
            - ``record_id`` -- (optional) file to store backend job ID's

        .. NOTE::

            The ``internal`` flag is meant for internal use only: it measures
            ``self._internal_ansatz`` when implementing the parameter-shift rule.

        :EXAMPLE:
            
            >>> from ionqvision.ansatze import VariationalAnsatz
            >>> from qiskit.circuit import Parameter
            >>> theta = Parameter("t")
            >>> ansatz = VariationalAnsatz(1)
            >>> _ = ansatz.rx(theta, 0)
            >>> ansatz.measure_ansatz([3.1416], shots=1e4)
            {'1': 10000}
            {'0': 518, '1': 482}
        """
        ansatz = self._internal_ansatz if internal else self
        ansatz = ansatz.decompose(COMPOSITE_GATES, reps=4)

        if isinstance(basis, str):
            if not len(basis) == ansatz.num_qubits:
                raise ValueError("Measurement basis must be specified for all qubits")
            ansatz = ansatz.compose(change_of_basis_circuit(basis))
        if isinstance(basis, QuantumCircuit):
            if not basis.num_qubits == ansatz.num_qubits:
                raise ValueError(r"Number of qubits in basis must match the number of qubits in self")
            ansatz.compose(basis)

        bound_qc = ansatz.assign_parameters(param_vals)
        if backend is None:
            backend = Aer.get_backend('aer_simulator')
        if "statevector" in backend.name:
            shots = 1
        else:
            bound_qc.measure_all()
        quantum_job = robust_backend_call(backend.run, (bound_qc,), {"shots": int(shots)})
        if record_id:
            with open(record_id, "a") as f:
                f.write(f"Job ID: {quantum_job.job_id()}. Submitted on {datetime.today()}.\n")
        if "statevector" in backend.name:
            counts = robust_backend_call(quantum_job.result).get_statevector().probabilities_dict()
        else:
            counts = robust_backend_call(quantum_job.result).get_counts()
        return counts

    def num_gates(self):
        """
        Count the number gates in ``self``.

        .. NOTE::

            This method does *not* decompose composite gates.

        :EXAMPLE:
            
            >>> from ionqvision.ansatze import VariationalAnsatz
            >>> from qiskit.circuit.library.standard_gates.xx_plus_yy import XXPlusYYGate
            >>> ansatz = VariationalAnsatz(3)
            >>> _ = ansatz.y(0)
            >>> _ = ansatz.barrier()
            >>> _ = ansatz.append(XXPlusYYGate(1), [0, 1], [])
            >>> _ = ansatz.cnot(0, 2)
            >>> ansatz.draw()
                 ┌───┐ ░ ┌───────────────┐     
            q_0: ┤ Y ├─░─┤0              ├──■──
                 └───┘ ░ │  (XX+YY)(1,0) │  │  
            q_1: ──────░─┤1              ├──┼──
                       ░ └───────────────┘┌─┴─┐
            q_2: ──────░──────────────────┤ X ├
                       ░                  └───┘
            >>> ansatz.num_gates()
            {'1q': 1, '2q': 2}
        """
        gate_ctr = defaultdict(int)
        for instruction in self:
            if not instruction.operation.name == "barrier":
                num_qubits = len(instruction.qubits)
                gate_ctr[str(num_qubits) + "q"] += 1
        return dict(gate_ctr)

    def rbs(self, theta, qubit1, qubit2):
        r"""
        Apply a Reconfigurable Beam Splitter (RBS) gate with parameter ``theta``
        onto ``qubit1`` and ``qubit2``.

        This gate acts as a *clockwise* rotation by an angle of ``theta``
        in the (real) $(\vert 01 \rangle, \vert 10 \rangle)$-plane of the
        subspace defined by the tensor product of the state spaces corresponding
        to ``qubit1`` and ``qubit2``. In other words, the following matrix
        describes the transformation defined by ``rbs(theta, 0, 1)``:

        .. math::

            \begin{bmatrix}
            1 & 0 & 0 & 1 \\
            0 & \cos(\theta) & -\sin(\theta) & 0 \\
            0 & -\sin(\theta) & \cos(\theta) & 0 \\
            1 & 0 & 0 & 1
            \end{bmatrix}.

        :EXAMPLE:

            >>> from ionqvision.ansatze import VariationalAnsatz
            >>> from qiskit.circuit import Parameter
            >>> ansatz = VariationalAnsatz(2)
            >>> _ = ansatz.x(0)
            >>> _ = ansatz.rbs(Parameter("t"), 0, 1)
            >>> ansatz.draw()
                 ┌───┐┌─────────┐
            q_0: ┤ X ├┤0        ├
                 └───┘│  RBS(t) │
            q_1: ─────┤1        ├
                      └─────────┘
            >>> expr = ansatz.get_symbolic_expression()
            >>> expr.simplify(); expr
            Matrix([
            [          0],
            [ 1.0*cos(t)],
            [-1.0*sin(t)],
            [          0]])
        """
        sub_circ = QuantumCircuit(2, name="RBS")
        sub_circ.s(0)
        sub_circ.s(1)
        sub_circ.h(1)
        sub_circ.cx(1, 0)
        sub_circ.ry(theta, 0)
        sub_circ.ry(theta, 1)
        sub_circ.cx(1, 0)
        sub_circ.h(1)
        sub_circ.sdg(1)
        sub_circ.sdg(0)
        return self.append(sub_circ.to_instruction(), [qubit1, qubit2])

    @property
    def symbols(self):
        """
        Get an (ordered) list of SymPy symbols in the symbolic expression of
        ``self``.

        .. NOTE::

            The list of symbols is computed in the parameter order fixed by
            :meth:`get_param_order`.

        :EXAMPLE:

            >>> from ionqvision.ansatze import VariationalAnsatz
            >>> from qiskit.circuit import Parameter
            >>> t, r = Parameter("t"), Parameter("r")
            >>> ansatz = VariationalAnsatz(1)
            >>> _ = ansatz.rx(t, 0)
            >>> _ = ansatz.ry(r, 0)
            >>> ansatz.draw()
               ┌───────┐┌───────┐
            q: ┤ Rx(t) ├┤ Ry(r) ├
               └───────┘└───────┘
            >>> ansatz.set_param_order([t, r])
            >>> ansatz.symbols
            [t, r]
        """
        if self._param_symbols is None or len(self._param_symbols) != len(self.get_param_order()):
            self.get_symbolic_expression()
        return [self._param_symbols[p] for p in self.get_param_order()]

    def to_file(self, path):
        with open(path, "wb") as f:
            qpy.dump(self, f)
