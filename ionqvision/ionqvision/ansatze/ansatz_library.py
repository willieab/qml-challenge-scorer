#
# IonQ, Inc., Copyright (c) 2024,
# All rights reserved.
# Use in source and binary forms of this software, without modification,
# is permitted solely for the purpose of activities associated with the IonQ
# Hackathon at Quantum Korea hosted by SKKU at Hotel Samjung and only during the
# June 21-23, 2024 duration of such event.
#
import numpy as np
import symengine as symeng

from math import ceil, log
from ionqvision.ansatze import VariationalAnsatz
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector
from qiskit.quantum_info import SparsePauliOp


class AngleEncoder(VariationalAnsatz):
    """
    Implement a quantum circuit for higher-order sparse angle encoding.

    INPUT:

        - ``num_qubits`` -- number of qubits
        - ``entanglement_depth`` -- (optional) number layers of entangling CNOT
          gates: for each ``k`` in ``range(entanglement_depth)``, use gates
          ``CNOT(j, j + k + 1)``.
        - ``param_prefix`` -- (optional) string prefix for named circuit
          parameters

    EXAMPLES::

        >>> from ionqvision.ansatze.ansatz_library import AngleEncoder
        >>> ansatz = AngleEncoder(4, entanglement_depth=3, param_prefix="y")
        >>> ansatz.draw()
             ┌────────────┐                              
        q_0: ┤ Ry(π*y[0]) ├──■──────────────■─────────■──
             ├────────────┤┌─┴─┐            │         │  
        q_1: ┤ Ry(π*y[1]) ├┤ X ├──■─────────┼────■────┼──
             ├────────────┤└───┘┌─┴─┐     ┌─┴─┐  │    │  
        q_2: ┤ Ry(π*y[2]) ├─────┤ X ├──■──┤ X ├──┼────┼──
             ├────────────┤     └───┘┌─┴─┐└───┘┌─┴─┐┌─┴─┐
        q_3: ┤ Ry(π*y[3]) ├──────────┤ X ├─────┤ X ├┤ X ├
             └────────────┘          └───┘     └───┘└───┘
    """
    def __init__(self, num_qubits, entanglement_depth=1, param_prefix="x"):
        super().__init__(num_qubits)

        x = ParameterVector(param_prefix, num_qubits)
        [self.ry(np.pi * xi, qbt) for qbt, xi in enumerate(x)]

        for k in range(entanglement_depth):
            top_qubit = 0
            while top_qubit < num_qubits - (k + 1):
                self.cx(top_qubit, top_qubit + k + 1)
                top_qubit += 1


class BrickworkLayoutAnsatz(VariationalAnsatz):
    r"""
    Construct a variational ansatz with a brickwork layout structure.

    By default, ``blk_sz`` independent parameters are assigned to each two-qubit
    "brick". However, a list of lists ``params`` for each gate may be supplied
    optionally. If used, ensure
    ``len(params) == ceil(num_layers * (num_qubits - 1) / 2))``.

    .. NOTE::

        A sub-class may override the :meth:`two_qubit_block` to define a custom
        "brick".

    A ``QuantumCircuit`` preparing the ``initial_state`` may be supplied
    optionally. In addition, a list of ``qubits`` may be passed, which indicates
    where the bricks should be laid.

    :EXAMPLE:
        
        >>> from ionqvision import ansatz_library
        >>> ansatz = ansatz_library.BrickworkLayoutAnsatz(3, 4)
        >>> ansatz.draw()
                ┌───────────┐                   ┌───────────┐                
        q_0: ─■─┤ Ry(θ0[0]) ├─────────────────■─┤ Ry(θ2[0]) ├────────────────
              │ ├───────────┤   ┌───────────┐ │ ├───────────┤   ┌───────────┐
        q_1: ─■─┤ Ry(θ0[0]) ├─■─┤ Ry(θ1[0]) ├─■─┤ Ry(θ2[0]) ├─■─┤ Ry(θ3[0]) ├
                └───────────┘ │ ├───────────┤   └───────────┘ │ ├───────────┤
        q_2: ─────────────────■─┤ Ry(θ1[0]) ├─────────────────■─┤ Ry(θ3[0]) ├
                                └───────────┘                   └───────────┘

    Alternatively, we may use absolute path imports.

    :EXAMPLE:

        >>> from ionqvision.ansatze.ansatz_library import BrickworkLayoutAnsatz
        >>> from qiskit.circuit import ParameterVector
        >>> params = [[t] for t in ParameterVector("t", 5)]
        >>> ansatz = BrickworkLayoutAnsatz(4, 3, params)
        >>> ansatz.draw()
                ┌──────────┐                  ┌──────────┐
        q_0: ─■─┤ Ry(t[0]) ├────────────────■─┤ Ry(t[3]) ├
              │ ├──────────┤   ┌──────────┐ │ ├──────────┤
        q_1: ─■─┤ Ry(t[0]) ├─■─┤ Ry(t[2]) ├─■─┤ Ry(t[3]) ├
                ├──────────┤ │ ├──────────┤   ├──────────┤
        q_2: ─■─┤ Ry(t[1]) ├─■─┤ Ry(t[2]) ├─■─┤ Ry(t[4]) ├
              │ ├──────────┤   └──────────┘ │ ├──────────┤
        q_3: ─■─┤ Ry(t[1]) ├────────────────■─┤ Ry(t[4]) ├
                └──────────┘                  └──────────┘
    """
    def __init__(self, num_qubits, num_layers, params=None, prefix=None, blk_sz=1, qubits=None, initial_state=None):
        super().__init__(num_qubits)
        if initial_state is not None:
            self.compose(initial_state, inplace=True)
            self.barrier()

        num_blocks = ceil(num_layers * (num_qubits - 1) / 2)
        if params is None:
            if prefix is None:
                prefix = "θ"
            params = [ParameterVector(prefix + str(j), blk_sz) for j in range(num_blocks)]
        else:
            if not len(params) == num_blocks:
                raise ValueError(f"params must have length ceil(num_layers * (num_qubits - 1) / 2)")

        param_it = iter(params)
        if qubits is None:
            qubits = list(range(num_qubits))
        for j in range(num_layers):
            top_idx = j % 2
            while top_idx < len(qubits) - 1:
                self.two_qubit_block(next(param_it), qubits[top_idx], qubits[top_idx+1])
                top_idx += 2

    def two_qubit_block(self, theta, q1, q2):
        """
        Add a single-parameter two-qubit "brick" to ``self``.

        Assumes ``theta`` is a list or ``ParameterVector`` of length 1.
        """
        self.cz(q1, q2)
        self.ry(theta[0], q1)
        self.ry(theta[0], q2)


class ButterflyOrthogonalAnsatz(VariationalAnsatz):
    def __init__(self, num_qubits, param_prefix="θ"):
        """
        Construct the so-called "Butterfly" orthogonal quantum layer as
        illustrated in Fig. 8 of "Quantum Vision Transformers" by Cherrat et al.

        :EXAMPLE:

            >>> from ionqvision.ansatze.ansatz_library import ButterflyOrthogonalAnsatz
            >>> ansatz = ButterflyOrthogonalAnsatz(4)
            >>> ansatz.draw()
                 ┌────────────┐               ░ ┌────────────┐ ░ 
            q_0: ┤0           ├───────────────░─┤0           ├─░─
                 │            │┌────────────┐ ░ │  RBS(θ[2]) │ ░ 
            q_1: ┤  RBS(θ[0]) ├┤0           ├─░─┤1           ├─░─
                 │            ││            │ ░ ├────────────┤ ░ 
            q_2: ┤1           ├┤  RBS(θ[1]) ├─░─┤0           ├─░─
                 └────────────┘│            │ ░ │  RBS(θ[3]) │ ░ 
            q_3: ──────────────┤1           ├─░─┤1           ├─░─
                               └────────────┘ ░ └────────────┘ ░ 
        """
        d = int(log(num_qubits, 2))
        if abs(log(num_qubits, 2) - d) > 1e-8:
            raise ValueError("num_qubits nums be a power of 2")

        super().__init__(num_qubits)
        theta = iter(ParameterVector(param_prefix, d * num_qubits//2))
        for depth in reversed(range(d)):
            for j in range(2**depth):
                for i in range(2**(d - depth - 1)):
                    offset = i*2**(depth + 1)
                    qubits = [offset + j, offset + j + 2**depth]
                    self.rbs(next(theta), *qubits)
            self.barrier()


class CrossOrthogonalAnsatz(VariationalAnsatz):
    def __init__(self, num_qubits, param_prefix="θ"):
        """
        Construct the so-called "Cross" orthogonal quantum layer as illustrated
        in Fig. 9 of "Quantum Vision Transformers" by Cherrat et al.

        :EXAMPLE:

            >>> from ionqvision.ansatze.ansatz_library import CrossOrthogonalAnsatz
            >>> ansatz = CrossOrthogonalAnsatz(6)
            >>> ansatz.draw()
                 ┌────────────┐                                          ┌────────────┐
            q_0: ┤0           ├──────────────────────────────────────────┤0           ├
                 │  RBS(θ[0]) │┌────────────┐              ┌────────────┐│  RBS(θ[8]) │
            q_1: ┤1           ├┤0           ├──────────────┤0           ├┤1           ├
                 └────────────┘│  RBS(θ[2]) │┌────────────┐│  RBS(θ[6]) │└────────────┘
            q_2: ──────────────┤1           ├┤0           ├┤1           ├──────────────
                               ├────────────┤│  RBS(θ[4]) │├────────────┤              
            q_3: ──────────────┤0           ├┤1           ├┤0           ├──────────────
                 ┌────────────┐│  RBS(θ[3]) │└────────────┘│  RBS(θ[5]) │┌────────────┐
            q_4: ┤0           ├┤1           ├──────────────┤1           ├┤0           ├
                 │  RBS(θ[1]) │└────────────┘              └────────────┘│  RBS(θ[7]) │
            q_5: ┤1           ├──────────────────────────────────────────┤1           ├
                 └────────────┘                                          └────────────┘
        """
        if num_qubits % 2:
            raise ValueError("num_qubits must be even")

        super().__init__(num_qubits)
        theta = iter(ParameterVector(param_prefix, 2*num_qubits - 3))
        for j in range(num_qubits-1):
            self.rbs(next(theta), j, j+1)
            if j != num_qubits - j - 2:
                self.rbs(next(theta), -2-j, -1-j)


class QAOAAnsatz(VariationalAnsatz):
    r"""
    Construct a (multi-angle) QAOA :class:`.VariationalAnsatz`.

    INPUT:

        - ``hamiltonian`` -- system Hamiltonian given as a Qiskit
          ``SparsePauliOp``
        - ``multi_angle`` -- (optional) boolean indicating whether to produce a
          multi-angle QAOA ansatz with an independent parameter on each
          rotation gate
        - ``depth`` -- (optional) number of ansatz layers.
        - ``initial_state`` -- (optional) non-variational ``QuantumCircuit``
          prepended to the QAOA ansatz; defaults to
          $\vert + \rangle^{\otimes n}$, with ``n = hamiltonian.num_qubits``
          when ``None`` is provided
        - ``rot`` -- (optional) axis of rotation in the entanglement layer

    :EXAMPLE:
        
        >>> from ionqvision import ansatz_library
        >>> from qiskit.quantum_info import SparsePauliOp
        >>> H = SparsePauliOp.from_list([("ZZ", 1), ("XI", -2), ("IZ", 3)])
        >>> ansatz = ansatz_library.QAOAAnsatz(H, multi_angle=False, depth=1, rot="Y")
        >>> ansatz.draw()
             ┌───┐ ░                 ┌──────────────┐ ░ ┌──────────────┐ ░ 
        q_0: ┤ H ├─░──■──────────────┤ Rz(6.0*θ[0]) ├─░─┤ Ry(2.0*θ[1]) ├─░─
             ├───┤ ░  │ZZ(2.0*θ[0]) ┌┴──────────────┤ ░ ├──────────────┤ ░ 
        q_1: ┤ H ├─░──■─────────────┤ Rx(-4.0*θ[0]) ├─░─┤ Ry(2.0*θ[1]) ├─░─
             └───┘ ░                └───────────────┘ ░ └──────────────┘ ░ 
    """
    def __init__(self, hamiltonian, multi_angle=True, depth=1, initial_state=None, rot="X"):
        n = hamiltonian.num_qubits
        super().__init__(n)
        if initial_state is None:
            [self.h(j) for j in range(n)]
        else:
            [self.append(instruction) for instruction in initial_state]
        self.barrier()
        
        rot_layer = SparsePauliOp.from_list([("".join(rot if q == j else "I" for q in range(n)), 1) for j in range(n)])
        herm_ops = list(hamiltonian) + list(rot_layer) if multi_angle else [hamiltonian, rot_layer]
        herm_ops *= depth
        eval_params = ParameterVector("θ", len(herm_ops))
        for j, layer in enumerate(herm_ops):
            for word, coeff in layer.label_iter():
                idx = [k for k, sig in enumerate(reversed(word)) if sig != "I"]
                if idx and coeff != 0:
                    ax = "".join(sig for sig in reversed(word) if sig != "I").lower()
                    rot = getattr(self, 'r' + ax)
                    rot(2*coeff.real * eval_params[j], *idx)
            self.barrier()


class QCNNAnsatz(VariationalAnsatz):
    r"""
    Implement the Quantum Convolutional Network Ansatz (QCNN) as described in
    :cite:t:`2019:qcnn`.

    The quasi-local unitary $U_i$'s are entangling two-qubit gates with $6$
    variational parameters.
    They are laid out in a brickwork pattern with ``filter_depth`` layers.

    The pooling operations are implemented by two-qubit controlled rotations,
    with $2$ variational parameters.

    The circuit starts with ``num_qubits`` active qubits and then half the
    remaining qubits are discarded after each pooling operation until only a
    single active qubit remains. This final qubit is measured and the result is
    used for binary classification.
    """
    class ConvolutionBrickwork(BrickworkLayoutAnsatz):
        """
        Implement the convolution filters for the :class:`.QCNNAnsatz`.
        """
        def __init__(self, num_qubits, num_layers, prefix=None, qubits=None, initial_state=None):
            super().__init__(num_qubits, num_layers, blk_sz=3, prefix=prefix, qubits=qubits, initial_state=initial_state)
        
        def two_qubit_block(self, theta, q1, q2):
            conv_op = QuantumCircuit(2, name="CONV")
            conv_op.ry(theta[0], 0)
            conv_op.ry(theta[1], 1)
            conv_op.rxx(theta[2], 0, 1)
            self.append(conv_op.to_instruction(), [q1, q2])

    class PoolingLayer(BrickworkLayoutAnsatz):
        """
        Implement the pooling layer for the :class:`.QCNNAnsatz`.
        """
        def __init__(self, num_qubits, prefix=None, qubits=None):
            super().__init__(num_qubits, 1, blk_sz=1, prefix=prefix, qubits=qubits)
    
        def two_qubit_block(self, theta, q1, q2):
            pool_op = QuantumCircuit(2, name="POOL")
            pool_op.crz(theta[0], 1, 0)
            self.append(pool_op.to_instruction(), [q1, q2])

    def __init__(self, num_qubits, filter_depth=2, initial_state=None):
        num_layers = int(log(num_qubits, 2))
        if abs(log(num_qubits, 2) - num_layers) > 1e-6:
            raise ValueError("num_qubits must be a power of 2")

        super().__init__(num_qubits)
        if initial_state is not None:
            self.compose(initial_state, inplace=True)

        for k in range(num_layers):
            qubits = list(range(0, num_qubits, 2**k))
        
            conv = QCNNAnsatz.ConvolutionBrickwork(num_qubits, filter_depth, prefix="C" + str(k), qubits=qubits)
            self.compose(conv, inplace=True)
            
            pool = QCNNAnsatz.PoolingLayer(num_qubits, prefix="P" + str(k), qubits=qubits)
            self.compose(pool, inplace=True)


class UnaryEncoder(VariationalAnsatz):
    def __init__(self, num_qubits, param_prefix="x"):
        r"""
        Construct a :class:`.UnaryEncoder` circuit from the given vector ``x``.

        The resulting circuit constructs the state
        $|x \rangle = \frac{1}{||x||} \sum_j |e_j \rangle$, with $e_j$ denoting
        the standard basis vector. In the Qiskit bit order, $e_j$ is the
        $2^(n - 1 - j)$th vector, with $n$ denoting ``len(x)``.

        .. NOTE::

            Assumes ``x`` is a unit vector in the non-negative orthant and
            ``len(x)`` is a power of 2.

        :EXAMPLE:

            >>> from ionqvision.ansatze.ansatz_library import UnaryEncoder
            >>> ansatz = UnaryEncoder(4)
            >>> ansatz.draw()
                                              ┌──────────────────────┐
            q_0: ─────────────────────────────┤0                     ├
                      ┌──────────────────────┐│  RBS(x[0],x[1],x[2]) │
            q_1: ─────┤0                     ├┤1                     ├
                      │                      │├──────────────────────┤
            q_2: ─────┤  RBS(x[0],x[1],x[2]) ├┤0                     ├
                 ┌───┐│                      ││  RBS(x[0],x[1],x[2]) │
            q_3: ┤ X ├┤1                     ├┤1                     ├
                 └───┘└──────────────────────┘└──────────────────────┘
        """
        d = int(log(num_qubits, 2))
        if not np.isclose(log(num_qubits, 2), d):
            raise ValueError("num_qubits must be a power of 2")

        symbols = list(symeng.symbols(f"x:{num_qubits-1}", real=True))
        xn = symeng.sqrt(1 - sum(x**2 for x in symbols))

        r = [symbols + [xn]]
        theta_symb = list()
        for layer in reversed(range(d)):
            layer_r, layer_theta = list(), list()
            for j in range(2**layer):
                rr = symeng.sqrt(r[-1][2*j]**2 + r[-1][2*j+1]**2)
                layer_r.append(rr)
                t = symeng.acos(r[-1][2*j] / rr)
                layer_theta.append(t)
            r.append(layer_r)
            theta_symb.append(layer_theta)

        num_params = sum(len(layer_params) for layer_params in theta_symb)
        theta = ParameterVector(param_prefix, num_params)
        param_symbols = {t: s for t, s in zip(theta, r[0])}

        super().__init__(num_qubits)
        self.x(-1)
        for j, layer_symbols in enumerate(reversed(theta_symb)):
            for i, symbol in enumerate(layer_symbols):
                q2 = -(1 + i * 2**(d-j))
                q1 = q2 - 2**(d-1-j)
                param = ParameterExpression(param_symbols, expr=symbol)
                self.rbs(param, q1, q2)
