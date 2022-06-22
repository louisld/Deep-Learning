from pickletools import optimize
import torch
from torch import nn
import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
import gzip

from neuralnetwork import NeuralNetwork

# Utilisation du GPU si disponible
device = "cuda" if torch.cuda.is_available() else "cpu"

# Chargement du jeu de données
fp = gzip.open('../fashion-mnist.pk.gz', 'rb')
allXtrain, allYtrain, Xtest, Ytest, classlist  = pickle.load(fp) 

Xtrain, Ytrain  = allXtrain[:20000].to(device), allYtrain[:20000].to(device)
Xvalid, Yvalid  = allXtrain[20000:30000].to(device), allYtrain[20000:30000].to(device)

# Initialisation du réseau neuronal
model = NeuralNetwork().to(device)

def train_loop(model, loss_fn, optimizer):
    """
    Boucle d'entraînement de l'algorithme
    avec utilisation de mini-batch.
    """
    Ntrain = Xtrain.shape[0]
    Nvalid = Xvalid.shape[0]

    idx = np.arange(Ntrain)
    batch_size = 200
    nbatch = int(Ntrain/batch_size)

    np.random.shuffle(idx)
    l = 0
    for bi in range(nbatch):
        ids = idx[bi*batch_size:(bi+1)*batch_size]
        images = Xtrain[ids]
        labels = Ytrain[ids]

        # Compute
        logprobs = model(images)
        loss = loss_fn(logprobs, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        l += loss.item()

    return l/nbatch

# Initialisation de la fonction de perte et de l'optimisateur
loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Boucle de l'algorithme
epochs = 30
train_losses = []

for t in range(epochs):
    print(f"Epoch {t+1}\n---------------")
    train_losses.append(train_loop(model, loss_fn, optimizer))
print("Done !")

plt.plot(train_losses)
plt.xlabel("epochs")
plt.ylabel("Pertes")
plt.show()