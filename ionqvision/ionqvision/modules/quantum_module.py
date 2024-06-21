#
# IonQ, Inc., Copyright (c) 2024,
# All rights reserved.
# Use in source and binary forms of this software, without modification,
# is permitted solely for the purpose of activities associated with the IonQ
# Hackathon at Quantum Korea hosted by SKKU at Hotel Samjung and only during the
# June 21-23, 2024 duration of such event.
#
import torch
import torch.autograd as autograd
import torch.nn as nn

from ionqvision.ansatze import VariationalAnsatz
from ionqvision.quantum_functions import HamiltonianEnergy
from ionqvision.quantum_function_category.hybrid_function import HybridFunction


class QuantumModule(nn.Module):
    """
    A PyTorch ``nn.Module`` that implements the execution of any
    :class:`.QuantumObjective` using the ``autograd`` engine.

    This allows the ``autograd`` engine to automatically differentiate
    the quantum objective using the parameter-shift rule.
    """
    def __init__(self, encoder, trainable_ansatz, obs, backend=None, shots=1000):
        super().__init__()
        self.encoder = encoder
        self.trainable_ansatz = trainable_ansatz
        self.backend = backend
        self.shots = shots

        inpts = [encoder.parameters]
        weights = trainable_ansatz.parameters

        self.layer_qc = VariationalAnsatz.from_quantum_circuit(encoder)
        self.layer_qc.barrier()
        self.layer_qc.compose(trainable_ansatz, inplace=True)

        self.quantum_features = obs
        features_qfun = [HamiltonianEnergy(ham, self.layer_qc) for ham in obs]

        self._torch_qfun = [TorchQuantumFunction(qfun, inpts, weights, backend, shots) for qfun in features_qfun]

        self.weights = torch.nn.Parameter(torch.rand(len(weights)))

    def forward(self, x):
        ret = [torch.stack([qfun(img, weights=self.weights) for qfun in self._torch_qfun]) for img in x]
        return torch.stack(ret).float()


class AutogradQuantumFunction(autograd.Function):
    """
    A class to model quantum functions that are trainable with PyTorch.
    This class is mostly a wrapper for :class:`.HybridFunction` that sets up
    the :meth:`forward` and :meth:`backward` methods expected by ``torch``; in
    other words, this class registers the evaluation of a
    :class:`.HybridFunction` in the ``autograd`` framework.

    .. NOTE::

        This class is *not* meant to be used directly; use
        :class:`.TorchQuantumFunction` or :class:`.QuantumModule` instead.
    """
    @staticmethod
    def forward(*args):
        *layer_inputs, weights, hybrid_func, backend, shots = args
        return torch.tensor(hybrid_func.fun(layer_inputs, weights, backend, shots))

    @staticmethod
    def setup_context(ctx, inputs, output):
        *layer_inputs, weights, hybrid_func, backend, shots = inputs
        ctx.save_for_backward(*layer_inputs, weights)

        ctx.hybrid_func = hybrid_func
        ctx.backend = backend
        ctx.shots = shots

    @staticmethod
    def backward(ctx, grad_output):
        *layer_inputs, weights = ctx.saved_tensors

        *dinputs, dweights = ctx.hybrid_func.grad(layer_inputs, weights, ctx.backend, ctx.shots)
        grad_inputs = [grad_output * torch.from_numpy(dinput) for dinput in dinputs]
        grad_weights = grad_output * torch.from_numpy(dweights)
        return *grad_inputs, grad_weights, None, None, None


class TorchQuantumFunction(HybridFunction):
    """
    A convenience class provided to call a quantum function that has been wrapped
    as a ``torch.autograd.Function``.
    """
    def __init__(self, quantum_function, layer_inputs, weights, backend=None, shots=1000):
        super().__init__(quantum_function, layer_inputs, weights)
        self.backend = backend
        self.shots = shots

    def __call__(self, *layer_inputs, weights):
        """
        Evaluate the wrapped :class:`.AutogradQuantumFunction`, and register the
        transformation in the computational graph.
        """
        return AutogradQuantumFunction.apply(*layer_inputs, weights, self, self.backend, self.shots)
