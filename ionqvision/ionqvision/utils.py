#
# IonQ, Inc., Copyright (c) 2024,
# All rights reserved.
# Use in source and binary forms of this software, without modification,
# is permitted solely for the purpose of activities associated with the IonQ
# Hackathon at Quantum Korea hosted by SKKU at Hotel Samjung and only during the
# June 21-23, 2024 duration of such event.
#
import logging
import matplotlib.pyplot as plt
import numpy as np

from collections import OrderedDict
from datetime import datetime
from matplotlib import cm
from os import getpid
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ionq.exceptions import IonQAPIError
from requests.exceptions import RequestException
from time import sleep
from urllib3.exceptions import HTTPError, PoolError

def change_of_basis_circuit(basis):
    r"""
    Construct a ``QuantumCircuit`` implementing a change of basis operation.

    INPUT:

        - ``basis`` -- a word in ``X``, ``Y``, and ``Z`` indicating the desired
          Pauli measurement basis for each qubit

    This method uses the Hadamard gate $H$ to diagonalize the Pauli-X gate, and
    the composition $S^\dagger H$ to diagonalize the Pauli-Y gate.

    .. NOTE::

        The ``basis`` string is interpreted in Qiskit bit order: ``basis[-1]``
        determines the measurement basis for qubit ``q_0``.

    :EXAMPLE:
        
        >>> from ionqvision.utils import change_of_basis_circuit
        >>> cob_qc = change_of_basis_circuit("XYZ")
        >>> cob_qc.draw()
              ░             
        q_0: ─░─────────────
              ░ ┌─────┐┌───┐
        q_1: ─░─┤ Sdg ├┤ H ├
              ░ └┬───┬┘└───┘
        q_2: ─░──┤ H ├──────
              ░  └───┘      

    We verify the resulting circuit indeed implements the appropriate unitary
    operation.

    :EXAMPLE:

        >>> from qiskit.quantum_info import Operator, SparsePauliOp
        >>> H = SparsePauliOp.from_list([("XYZ", 1)]).to_matrix()
        >>> parity = [(-1)**sum(map(int, f"{x:b}")) for x in range(2**3)]
        >>> cob_qc = change_of_basis_circuit("XYZ")
        >>> V = Operator(cob_qc).data
        >>> np.allclose(V @ H @ V.conj().T, np.diag(parity))
        True
    """
    qubit_basis_change = {"X": ["h"], "Y": ["sdg", "h"], "Z": []}
    diagonalizer = QuantumCircuit(len(basis))
    diagonalizer.barrier()
    for j, XYZ in enumerate(reversed(basis)):
        for gate in qubit_basis_change[XYZ]:
            getattr(diagonalizer, gate)(j)
    return diagonalizer
    

def cvar_moments(values, counts, alpha=1.0):
    r"""
    Compute the CVaR expectation and variance of the given ``values``, sorted
    according to ``counts``.

    The parameter ``0.0 <= alpha <= 1.0`` indicates the fraction of samples to
    consider in the expectation calculation. Precisely, let $K$ denote
    ``counts.sum()`` and let $v_1, \ldots, v_K$ denote the entries of ``values``, 
    sorted in non-decreasing order, and with ``values[k]`` written with
    multiplicity ``counts[k]``. Then the CVaR expectation is given by
    $$
    \frac{1}{\lceil \alpha K \rceil} \sum_{k = 0}^{\lceil \alpha K \rceil} v_k.
    $$
    
    In :cite:t:`2020:Barkoutsos`, the authors provide empirical evidence
    suggesting that CVaR leads to faster convergence for various optimization
    problems. 

    .. NOTE:: 

        Both ``values`` and ``counts`` must be 1D-NumPy arrays. In addition, 
        they must be provided in the same order, so that ``counts[j]``
        describes the frequency of ``values[j]``.

    :EXAMPLE:
        
        >>> from ionqvision.utils import cvar_moments
        >>> values = np.array([1, 2, 3])
        >>> counts = np.array([3, 3, 4])
        >>> mu, _ = cvar_moments(values, counts)
        >>> mu == np.average(values, weights=counts)
        True
        >>> cvar_moments(values, counts, alpha=0.5)
        (1.4, 0.24000000000000044)
    """
    if np.isclose(alpha, 0.):
        return values.min(), 0.
    if np.isclose(alpha, 1.):
        shots = counts.sum()
        moments = [np.dot(values**p, counts) / shots for p in [1, 2]]
    else:
        sort_idx = np.argsort(values)
        sorted_counts, sorted_energies = counts[sort_idx], values[sort_idx]
        cdf = sorted_counts.cumsum()
        num_selected = np.ceil(alpha * cdf[-1])
        last_idx = np.argmin(cdf < num_selected)

        sorted_counts[last_idx] -= cdf[last_idx] - num_selected
        assert np.isclose(sorted_counts[:last_idx+1].sum(), num_selected)

        moments = [sorted_energies[:last_idx+1]**p @ sorted_counts[:last_idx+1] / num_selected for p in [1, 2]]
    return moments[0], moments[1] - moments[0]**2


def fast_ising_energies(ising_ham, states=None):
    """
    Quickly obtain eigenvalues of an Ising Hamiltonian corresponding to the
    given ``states``. 

    If ``states`` is ``None``, method returns all the eigenvalues.

    :EXAMPLE:
        
        >>> from ionqvision.utils import fast_ising_energies
        >>> from qiskit.quantum_info import SparsePauliOp
        >>> hamiltonian = SparsePauliOp.from_list([("IZ", 1), ("ZI", 2), ("ZZ", -3)])
        >>> states = np.array([[0, 1], [1, 0]])
        >>> fast_ising_energies(hamiltonian, states)
        array([4., 2.])
        >>> fast_ising_energies(hamiltonian)
        array([ 0.,  4.,  2., -6.])

    This method complies with the Qiskit bit order::
        
        >>> from qiskit.quantum_info import Operator
        >>> Operator(hamiltonian).data
        array([[ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
               [ 0.+0.j,  4.+0.j,  0.+0.j,  0.+0.j],
               [ 0.+0.j,  0.+0.j,  2.+0.j,  0.+0.j],
               [ 0.+0.j,  0.+0.j,  0.+0.j, -6.+0.j]])
    """
    if isinstance(ising_ham, SparsePauliOp):
        paulis = np.array([list(ops) for ops, _ in ising_ham.label_iter()]) != "I"
        coeffs = ising_ham.coeffs.real
        n = ising_ham.num_qubits
    else:
        paulis, coeffs = ising_ham
        n = len(paulis[0])
    if states is None:
        a = np.arange(2**n, dtype=int)[:, np.newaxis]
        b = np.arange(n, dtype=int)[np.newaxis, ::-1]
        states = np.array(2**b & a > 0, dtype=int)
    evals = (-1) ** (states @ paulis.T) @ coeffs
    return evals


def most_likely_states(counts, num_states=1, reverse_bit_order=False):
    """
    Extract the top ``num_states`` most probable outcomes from the ``counts`` 
    dictionary of ``(state, count)`` key-value pairs.

    Returns an :class:`OrderedDict` of ``(state, probability)`` pairs ranked by
    **decreasing** likelihood.

    By default, the states are returned in Qiskit bit-order. However, the order
    can be reversed using the ``reverse_bit_order`` flag.

    :EXAMPLE:
        
        >>> from ionqvision.utils import most_likely_states
        >>> counts = {"001": 3, "110": 7, "010": 1, "111": 5}
        >>> most_likely_states(counts, num_states=2)
        OrderedDict([('110', 0.4375), ('111', 0.3125)])
        >>> most_likely_states(counts, num_states=2, reverse_bit_order=True)
        OrderedDict([('011', 0.4375), ('111', 0.3125)])
    """
    shots = sum(counts.values())
    ranked = sorted(counts, key=lambda k: counts[k], reverse=True)
    if reverse_bit_order:
        return OrderedDict({state[::-1]: counts[state] / shots for state in ranked[:num_states]})
    return OrderedDict({state: counts[state] / shots for state in ranked[:num_states]})


def plot_cost_landscape(cost_fun, x_lims, y_lims, num_samples=50):
    r"""
    Plot the 3D surface described by the given ``cost_fun`` of two variational
    parameters by computing ``num_samples**2`` in the rectangle 
    ``x_lims`` $\times ``y_lims``.

    OUTPUT:

        - ``Z`` -- NumPy array describing the cost discretized surface
        - ``fig`` -- 3D PyPlot ``Figure`` with the plotted surface

    For example, we can inspect the cost landscape of the objective function of
    a Knapsack problem with respect to the standard QAOA ansatz. To do this,
    first we set up a :class:`.KnapsackSolver`::
        
        >>> from ionqvision.solvers import KnapsackSolver
        >>> values = [4, 5, 3, 7]
        >>> weights = [2, 3, 1, 4]
        >>> capacity = 5
        >>> solver = KnapsackSolver(values, weights, capacity, depth=1, multi_angle=False)

    Now we plot the solver objective as a function of the variational circuit
    parameters::
        
        >>> from ionqvision.utils import plot_cost_landscape
        >>> cost_fun = solver.objective.fun
        >>> Z, fig = plot_cost_landscape(cost_fun, [-np.pi, np.pi], [-np.pi/2, np.pi/2])

    Displaying ``fig`` results in the following image.

    .. image:: media/cost_landscape_ex_dark.png
      :class: only-dark
      :align: center
      :width: 600
      :alt: 3D surface plot of the Knapsack problem's variational objective

    .. image:: media/cost_landscape_ex_light.png
      :class: only-light
      :align: center
      :width: 600
      :alt: 3D surface plot of the Knapsack problem's variational objective
    """
    coords = np.meshgrid(*[np.linspace(*lims) for lims in [x_lims, y_lims]])
    Z = np.zeros_like(coords[0])
    for loc in np.ndindex(coords[0].shape):
        Z[loc] = complex(cost_fun([c[loc] for c in coords])).real

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(*coords, Z, cmap=cm.coolwarm)
    ax.set_xlim(reversed(x_lims))
    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$\theta_1$")
    ax.set_title("Variational cost landscape")
    return Z, fig


def robust_backend_call(fn, args=(), kwargs={}, wait_time=100, max_attempts=25):
    """
    Try to evaluate ``fn(*args, **kwargs)`` every ``wait_time`` seconds, up to
    ``max_attempt`` times, excepting any network and server errors. 

    Precisely, this method provides a means of calling the quantum backend
    method ``fn`` that is robust against any 
    :class:`reqests.exceptions.RequestException`'s, any
    :class:`urllib3.exceptions.HTTPError`'s, any
    :class:`urllib3.exceptions.PoolError`'s, and any 
    :class:`qiskit_ionq.exceptions.IonQAPIError`'s thrown.

    :EXAMPLE:
        
        >>> from ionqvision.utils import robust_backend_call
        >>> from qiskit_aer import Aer
        >>> simulator = Aer.get_backend('aer_simulator', seed=23)
        >>> from qiskit import QuantumCircuit
        >>> qc = QuantumCircuit(1)
        >>> _ = qc.h(0)
        >>> qc.measure_all()
        >>> quantum_job = robust_backend_call(simulator.run, args=(qc,), kwargs={"shots": 100})
        >>> result = robust_backend_call(quantum_job.result)
        {'0': 46, '1': 54}
    """
    suffix = {2: "nd", 3: "rd"}
    for k in range(max_attempts):
        try:
            if k > 0:
                print(f"Trying to execute {fn} for the {k+1}{suffix.get(k+1, 'th')} time...")
            res = fn(*args, **kwargs)
            return res
        except (HTTPError, IonQAPIError, PoolError, RequestException) as e:
            logging.warning(f"\nProcess {getpid()} ran into {type(e)}:{e} at {datetime.today()}.") 
            sleep(wait_time)
            continue
    raise RuntimeError(f"Robust backend call {fn} failed on process {getpid()}")
    