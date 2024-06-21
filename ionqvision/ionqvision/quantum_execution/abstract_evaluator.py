#
# IonQ, Inc., Copyright (c) 2024,
# All rights reserved.
# Use in source and binary forms of this software, without modification,
# is permitted solely for the purpose of activities associated with the IonQ
# Hackathon at Quantum Korea hosted by SKKU at Hotel Samjung and only during the
# June 21-23, 2024 duration of such event.
#
from abc import ABC, abstractmethod
from typing import Literal, Sequence, Tuple

class AbstractEvaluator(ABC):
    """
    An abstract class to manage the end-to-end evaluation of a
    :class:`.QuantumObjective`.
    """
    @abstractmethod
    def eval(
        self,
        quantum_objective,
        method: Literal["fun", "grad"],
        param_vals: Sequence[float],
        shots: int,
        cvar_alpha: float
        ) -> Tuple[float | Sequence[float], dict]:
        """
        Evaluate the chosen ``method`` (``fun`` or ``grad``) of the given
        ``quantum_objective`` at the point specificied by ``param_values`` and using
        the given number of ``shots``.

        In addition, this method computes a CVaR expectation value, with parameter
        ``cvar_alpha`` in the interval ``[0, 1]``; for details, see
        :meth:`ionqvision.utils.cvar_moments`.

        OUTPUT:

            Returns a pair whose first element is the result of the function or
            gradient evaluation and the second is a payload dictionary used by any
            object supporting the :class:`.AbstractLogger` interface to record
            optimization progress.
        """
        pass
