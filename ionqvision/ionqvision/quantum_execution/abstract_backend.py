#
# IonQ, Inc., Copyright (c) 2024,
# All rights reserved.
# Use in source and binary forms of this software, without modification,
# is permitted solely for the purpose of activities associated with the IonQ
# Hackathon at Quantum Korea hosted by SKKU at Hotel Samjung and only during the
# June 21-23, 2024 duration of such event.
#
from abc import ABC, abstractmethod
from datetime import datetime
from qiskit import QuantumCircuit
from typing import List, Tuple


class AbstractBackend(ABC):
    @abstractmethod
    def name(self):
        """
        Get the name of the ``backend``.
        """
        pass

    @abstractmethod
    def run(
        self,
        bound_circuits: List[QuantumCircuit],
        shots: int = 1000
        ) -> Tuple[List[dict], str, datetime]:
        """
        Execute the list of ``bound_circuits`` on the quantum backend target
        ``self`` and return a list of measurement results, a quantum job ID,
        and a job submission time stamp.
        """
        pass
