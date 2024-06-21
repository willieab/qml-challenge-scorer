Quantum-classical PyTorch modules
=================================

Classifier
----------

.. automodule:: ionqvision.modules.binary_classifier
   :members:
   :show-inheritance:

Quantum module
--------------

This module implements the `"quantum layer" <https://willieab.github.io/qml-challenge-scorer/challenge-description#quantum-layer>`_ in our hybrid classification pipeline.

.. autoclass:: ionqvision.modules.quantum_module.QuantumModule
   :members:
   :show-inheritance:

Trainable module
----------------

For convenience, we have provided a `torch` module implementing a basic training loop.

.. automodule:: ionqvision.modules.trainable_module
   :members:
   :show-inheritance: