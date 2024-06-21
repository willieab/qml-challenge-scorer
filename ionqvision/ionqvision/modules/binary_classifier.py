#
# IonQ, Inc., Copyright (c) 2024,
# All rights reserved.
# Use in source and binary forms of this software, without modification,
# is permitted solely for the purpose of activities associated with the IonQ
# Hackathon at Quantum Korea hosted by SKKU at Hotel Samjung and only during the
# June 21-23, 2024 duration of such event.
#
import tempfile
import shutil
from typing import List

import git
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle
from qiskit.quantum_info import SparsePauliOp
import torch
import torch.nn as nn
import torchvision as tv

from ionqvision.ansatze import VariationalAnsatz
from ionqvision.modules import QuantumModule
from ionqvision.modules.trainable_module import TrainableModule


MODELS_DIR = "trained_models"


class BinaryMNISTClassifier(TrainableModule):
    """
    A hybrid quantum-classical classifier for distinguishing between two types
    of handwritten digits in the MNIST database.

    The model architecture is detailed in the challenge description.
    """
    def __init__(
            self,
            encoder: VariationalAnsatz,
            trainable_ansatz: VariationalAnsatz,
            obs: List[SparsePauliOp],
            backend=None,
            shots: int=1000,
            dropout_prob: float=0.5
        ):
        self.num_pc = 6

        height, width = 28, self.num_pc
        in_dim = height * width

        super().__init__()
        self.latent_vec_encoder = nn.Sequential(*[
            nn.Linear(in_dim, trainable_ansatz.num_qubits), 
            nn.Dropout(dropout_prob),
            nn.Sigmoid()
        ])
        self.quantum_layer = QuantumModule(encoder, trainable_ansatz, obs, backend, shots)
        self.prediction_head = nn.Sequential(*[
            nn.Linear(len(obs), 1), 
            nn.Dropout(dropout_prob),
            nn.Sigmoid()
        ])

    def forward(self, x):
        x = x.reshape((x.shape[0], -1))
        
        x = self.latent_vec_encoder(x)
        x = self.quantum_layer(x)
        x = self.prediction_head(x)
        return x.squeeze()

    @classmethod
    def load_model(self, path):
        with tempfile.TemporaryDirectory() as tempdir:
            shutil.unpack_archive(path, tempdir, "zip") 

            encoder = VariationalAnsatz.from_file(f"{tempdir}/encoder.qpy")
            trainable_ansatz = VariationalAnsatz.from_file(f"{tempdir}/ansatz.qpy")
            with open(f"{tempdir}/observables.pkl", "rb") as f:
                features = pickle.load(f)
            features = list(SparsePauliOp.from_list(obs) for obs in features)

            model = BinaryMNISTClassifier(encoder, trainable_ansatz, features)
            model.load_state_dict(torch.load(f"{tempdir}/model_weights.pt"))
        return model

    def get_train_test_set(self, train_size=1_000, test_size=100):
        """
        Get sample sets of 0-1 MNIST images of size `train_size` and
        `test_size` for training and testing your model.

        .. NOTE:

            This method compresses the raw images by projecting them onto their
            leading `self.num_pc` principal components using PCA.

            Make sure you train your model on these compressed images!
        """
        mnist = self.load_binary_mnist()

        for split, sz in zip(["train", "test"], [train_size, test_size]):
            idx = torch.randperm(len(mnist[split]))[:sz]
            mnist[split].data, mnist[split].targets = mnist[split].data[idx].float(), mnist[split].targets[idx]

            _, _, V = torch.pca_lowrank(mnist[split].data, q=self.num_pc, niter=10)
            mnist[split].data = torch.matmul(mnist[split].data, V)
        return mnist["train"], mnist["test"]

    def load_binary_mnist(self):
        """
        Load all ``DIGIT1`` and ``DIGIT2`` images of the MNIST dataset.

        OUTPUT:
            
            A dictionary with keys `"train"` and `"test"` mapping to
            corresponding `torchvision.datasets` objects.
        """
        DIGIT1 = 0
        DIGIT2 = 1

        preproc = tv.transforms.Compose([
            tv.transforms.ToTensor(),
        ])
        mnist = dict()
        for split in ["train", "test"]:
            mnist[split] = tv.datasets.MNIST("./mnist-" + split, train=(split == "train"), download=True, transform=preproc)
        
            idx = (mnist[split].targets == DIGIT1) | (mnist[split].targets == DIGIT2)
            mnist[split].data, mnist[split].targets = mnist[split].data[idx], mnist[split].targets[idx]

            mnist[split].targets = mnist[split].targets != DIGIT1
        return mnist

    def save_model(self, path="", path_to_repo=None):
        """
        Serialize `self` and save ZIP archive to `path`.

        When no `path` is provided, the file name is generated automatically
        using the current commit's hash value.
        """
        if Path(path).is_dir() or not path:
            repo = git.Repo(path_to_repo, search_parent_directories=True)
            sha = repo.head.object.hexsha
            team = repo.active_branch.name
            path = Path(path).joinpath(f"model_{team}_{sha}")

        with tempfile.TemporaryDirectory() as tempdir:
            torch.save(self.state_dict(), f"{tempdir}/model_weights.pt")
            self.quantum_layer.encoder.to_file(f"{tempdir}/encoder.qpy")
            self.quantum_layer.trainable_ansatz.to_file(f"{tempdir}/ansatz.qpy")
            with open(f"{tempdir}/observables.pkl", "wb") as f:
                pickle.dump([obs.to_list() for obs in self.quantum_layer.quantum_features], f)
            shutil.make_archive(path, "zip", tempdir)
        return f"{path}.zip"

    def submit_model_for_grading(self, path_to_repo=None):
        """
        Submit `self` for grading.

        .. NOTE:

            This method stores a ZIP archive describing `self` in the 
            `trained_models` directory and it pushes a new `git` commit to the
            remote repository.
        """
        Path(MODELS_DIR).mkdir(exist_ok=True)
        path = self.save_model(MODELS_DIR, path_to_repo)
        repo = git.Repo(path_to_repo, search_parent_directories=True)
        repo.git.add(MODELS_DIR)
        repo.index.commit(f"Team {repo.active_branch} deploying new model")
        origin = repo.remote(name="origin")
        branch = repo.active_branch.name
        origin.push(refspec=f"{branch}:{branch}").raise_if_error()
        print(f"Success! Submitted {path} for grading!")

    def visualize_batch(self, batch=None):
        """
        Visualize a batch of images using PyPlot.

        If no `batch` is given, generate a random batch of 20 training images.
        """
        if batch is None:
            mnist = self.load_binary_mnist()
            batch, _ = next(iter(torch.utils.data.DataLoader(mnist["train"], batch_size=32, shuffle=True)))

        if len(batch.shape) == 3:
            batch = batch[:, None, :, :]
        plt.imshow(np.transpose(tv.utils.make_grid(batch)/2 + 0.5, (1, 2, 0)))
