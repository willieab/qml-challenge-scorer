#
# IonQ, Inc., Copyright (c) 2024,
# All rights reserved.
# Use in source and binary forms of this software, without modification,
# is permitted solely for the purpose of activities associated with the IonQ
# Hackathon at Quantum Korea hosted by SKKU at Hotel Samjung and only during the
# June 21-23, 2024 duration of such event.
#
from ionqvision.utils import fast_ising_energies
from .ising_energy_from_oracle import IsingEnergyFromOracle


class IsingEnergy(IsingEnergyFromOracle):
    r"""
    Compute the expected energy of the Ising Hamiltonian ``ising_ham``, given
    as a Qiskit :class:`SparsePauliOp`, by implementing its ``eigenvalues`` 
    using fast NumPy array products.

    For example, consider the Hamiltonian:
    $$
    \mathcal{H} = a \sigma^Z_1 + b \sigma^Z_2 + c \sigma^Z_1 \sigma^Z_2 + d
    $$
    on $2$-qubits, for some coefficients $(a, b, c, d)$.

    In addition, fix $U(\theta) = e^{-i \theta (\sigma^X_1 + \sigma^X_2)}$ and
    consider the variational quantum state defined by 
    $$
    \vert \psi(\theta)\rangle = U(\theta) \sigma^X_1 \vert 0 \rangle
    = e^{-i \theta (\sigma^X_1 + \sigma^X_2)} \vert 10\rangle.
    $$

    Some algebra shows that the expected observed energy is given by
    $$
    E(\theta) = \langle \psi(\theta) \vert \mathcal{H} \vert \psi(\theta)\rangle
    = (b - a) \cos(2\theta) - c \cos^2(2\theta) + d,
    $$
    so that its gradient with respect to the variational parameter $\theta$ is 
    $$
    \frac{dE}{d\theta}(\theta) = 2(a - b) \sin(2\theta) + 2 c \sin(4\theta).
    $$
    
    We reproduce these calculations as follows.

    :EXAMPLE:

        >>> from ionqvision.ansatze.ansatz_library import QAOAAnsatz
        >>> from ionqvision.quantum_functions import IsingEnergy
        >>> from qiskit import QuantumCircuit
        >>> from qiskit.quantum_info import SparsePauliOp
        >>> a, b, c, d = (-1, -1/3, 1.5, 50)
        >>> ising_ham = SparsePauliOp.from_list([('IZ', a), ('ZI', b), ('ZZ', c), ('II', d)])
        >>> initial_state = QuantumCircuit(2)
        >>> _ = initial_state.x(0)
        >>> ansatz = QAOAAnsatz(triv, multi_angle=False, initial_state=initial_state)
        >>> energy = IsingEnergy(ising_ham, ansatz)
        >>> energy.fun_symb()
        47.8333333333333*sin(1.0*θ[1])**4 + 103.0*sin(1.0*θ[1])**2*cos(1.0*θ[1])**2 + 49.1666666666667*cos(1.0*θ[1])**4
        >>> energy.grad_symb()
        Matrix([[-14.6666666666667*sin(1.0*θ[1])**3*cos(1.0*θ[1]) + 9.33333333333334*sin(1.0*θ[1])*cos(1.0*θ[1])**3]])

    We leave it to the reader to check that the two analytical expressions obtained are
    indeed equivalent! See :class:`.VariationalAnsatz` for setting up the ``ansatz``.
    """
    def __init__(self, ising_ham, ansatz):
        self.ising_ham = ising_ham
        super().__init__(ansatz)

    def eigenvalues(self, states=None):
        """
        Compute the Hamiltonian eigenvalues corresponding to the given ``states``.
        """
        return fast_ising_energies(self.ising_ham, states)    
