#
# IonQ, Inc., Copyright (c) 2024,
# All rights reserved.
# Use in source and binary forms of this software, without modification,
# is permitted solely for the purpose of activities associated with the IonQ
# Hackathon at Quantum Korea hosted by SKKU at Hotel Samjung and only during the
# June 21-23, 2024 duration of such event.
#
from datetime import datetime
from ionqvision.ansatze.variational_ansatz import COMPOSITE_GATES
from ionqvision.utils import robust_backend_call
from qiskit.compiler import transpile
from qiskit.providers.backend import Backend as BuiltInQiskitBackend
from qiskit_aer import Aer
from .abstract_backend import AbstractBackend


class QiskitBackend(AbstractBackend):
    def __init__(self, target: BuiltInQiskitBackend = None):
        self._target = Aer.get_backend("aer_simulator") if target is None else target

    def name(self):
        """
        Get the name of this backend.
        """
        return self._target.name

    def run(self, bound_circuits, shots=1000):
        """
        Execute the list of ``bound_circuits`` and obtain a list of circuit
        measurements.
        """
        bound_circuits = [qc.decompose(COMPOSITE_GATES, reps=4) for qc in bound_circuits]

        if "statevector" in self.name():
            shots = 1
        else:
            [circuit.measure_all() for circuit in bound_circuits]
        quantum_job = robust_backend_call(self._target.run, (bound_circuits,), {"shots": int(shots)})
        submission_time = datetime.today()

        res = robust_backend_call(quantum_job.result).get_counts()
        if isinstance(res, dict):
            res = [res]
        return res, quantum_job.job_id(), submission_time
