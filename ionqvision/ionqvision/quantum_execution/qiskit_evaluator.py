#
# IonQ, Inc., Copyright (c) 2024,
# All rights reserved.
# Use in source and binary forms of this software, without modification,
# is permitted solely for the purpose of activities associated with the IonQ
# Hackathon at Quantum Korea hosted by SKKU at Hotel Samjung and only during the
# June 21-23, 2024 duration of such event.
#
from .abstract_evaluator import AbstractEvaluator
from .qiskit_backend import QiskitBackend

class QiskitEvaluator(AbstractEvaluator):
    def __init__(self, backend: QiskitBackend):
        self._backend = backend

    def eval(self, quantum_objective, method, param_vals, shots, cvar_alpha):
        """
        Evaluate the chosen ``method`` of the given ``quantum_objective`` at the
        point specified by the ``param_values`` and using the given number of
        ``shots`` and the ``cvar_alpha``.
        """
        circuits = getattr(quantum_objective, "_prepare_quantum_execution_" + method)(param_vals)
        measurements, job_id, submission_time = self._backend.run(circuits, shots)

        res, var = getattr(quantum_objective, "_" + method)(measurements, cvar_alpha)

        metadata = {
            method: res,
            "job_id": job_id,
            "submission_time": submission_time,
            "params": param_vals,
            "shots": round(sum(measurements[0].values())),
            "var": var,
            "latest_counts": measurements
        }
        return res, metadata
