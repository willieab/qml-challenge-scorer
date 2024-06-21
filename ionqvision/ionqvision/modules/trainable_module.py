#
# IonQ, Inc., Copyright (c) 2024,
# All rights reserved.
# Use in source and binary forms of this software, without modification,
# is permitted solely for the purpose of activities associated with the IonQ
# Hackathon at Quantum Korea hosted by SKKU at Hotel Samjung and only during the
# June 21-23, 2024 duration of such event.
#
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from functools import reduce
from time import time



class TrainableModule(nn.Module):
    def __init__(self):
        super(TrainableModule, self).__init__()
        self.loss_per_epoch  = list()
        self.train_acc = list()
        self.test_acc = list()
    
    def train_module(self, train_set, test_set, train_config={}, criterion=nn.functional.binary_cross_entropy,):
        """
        Basic training loop.
    
        INPUT: 
    
            - ``train_set`` -- training data ``torchvision.datasets``
            - ``test_set`` -- test data ``torchvision.datasets``
            - ``criterion`` -- (optional) optimization objective. Defaults to ``nn.CrossEntropyLoss``
            - ``train_config`` -- (optional) configuration parameters. In addition to keys recognized
              by ``torch.optim.Adam``, ``epochs=20,`` ``clip_grad=False,`` and 
              ``log_interval=500`` are also supported.
        """
        config = train_config.copy()
        epochs = config.pop("epochs", 20)
        clip_grad = config.pop("clip_grad", False)
        log_interval = config.pop("log_interval", 500)
        batch_size = config.pop("batch_size", 50)
        optimizer = torch.optim.Adam(self.parameters(), **config)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.1)
        
        self.train()
        
        for epoch in range(epochs):
            t0 = time()
            epoch_loss = 0.
            log_ctr = 0
            batches = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
            for k, (batch, labels) in enumerate(batches):
                tf0 = time()
                loss = criterion(self(batch), labels.float())
                tff = time()
        
                optimizer.zero_grad()
                tb0 = time()
                loss.backward()
                tbf = time()
        
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                optimizer.step()
        
                epoch_loss += loss.item()
                if k % log_interval == log_interval - 1:
                    log_ctr += 1
                    lr = scheduler.get_last_lr()[0]
                    curr_loss = epoch_loss / (log_ctr * log_interval)
                    train_acc = self.compute_accuracy(train_set)
                    test_acc = self.compute_accuracy(test_set)
                    per_batch = (time() - t0)
                    print(f"epoch: {len(self.test_acc)+1:3d} | loss: {curr_loss:5.3f}")
                    print(f"lr: {lr:5.4f} | processed {k+1:5d}/{len(batches):5d} batches per epoch in {per_batch:4.2f}s ({tff-tf0:4.2f}s forward / {tbf-tb0:4.2f}s backward)")
                    print(f"Model achieved {100*train_acc:5.3f}%  accuracy on TRAIN set.")
                    print(f"Model achieved {100*test_acc:5.3f}%  accuracy on TEST set.\n")
        
            self.loss_per_epoch.append(epoch_loss / len(batches))
            self.train_acc.append(train_acc)
            self.test_acc.append(test_acc)

        self.train(mode=False)

    def compute_accuracy(self, dataset):
        """
        Compute the accuracy of ``self`` with respect to the ``dataset``, given
        as a ``torchvision.datasets`` object.
        """
        self.eval()
        with torch.no_grad():
            data, labels = dataset.data, dataset.targets
            y_pred = self(data).round()
            num_correct = (y_pred == labels).sum().item()
        return num_correct / data.shape[0]
    
    def plot_training_progress(self, outfile=""):
        """
        Plot the convergence of the loss function, together with the training and test set
        accuracy.
    
        Optionally store results in ``outfile``.
        """
        xx = range(1, len(self.test_acc) + 1)
        plt.plot(xx, self.loss_per_epoch, 'r', marker='o', markersize=8, label="Training loss")
        plt.plot(xx, self.train_acc, label="Training accuracy")
        plt.plot(xx, self.test_acc, label="Test accuracy")
        plt.legend()
        plt.title("Training progress")
        plt.xlabel("Epoch")
        if outfile:
            plt.savefig(outfile)
    
    def num_params(self):
        """
        Count the total number of parameters in a self.
        """
        prod = lambda t: reduce(operator.mul, t, 1)
        return sum(prod(p.shape) for p in self.parameters())
