import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import gzip
import pickle
from torchsummary import summary

from neuralnetwork import Sound, SoundDeep

# Utilisation du GPU si disponible
device = "cuda" if torch.cuda.is_available() else "cpu"

# Chargement des données d'entraînement
with gzip.open('../speech_commands_train.pck.gz', 'rb') as f:
    Xtrain, Ytrain, labels = pickle.load(f, encoding='latin1')
Xtrain, Ytrain = Xtrain.to(device), Ytrain.to(device)

#Chargement des données de test
with gzip.open('../speech_commands_test.pck.gz', 'rb') as f:
    Xtest, Ytest, labels_test = pickle.load(f, encoding='latin1')
Xtest, Ytest = Xtest.to(device), Ytest.to(device)

model = SoundDeep().to(device)
summary(model, (1, 16000))

def train_loop(model, loss_fn, optimizer, valid_accuracies):
    """
    Boucle d'entraînement de l'algorithme
    avec utilisation de mini-batch.
    """
    Ntrain = Xtrain.view(20000, 16000).shape[0]
    NValid = Xtest.shape[0]

    idx = np.arange(Ntrain)
    batch_size = 1000
    nbatch = int(Ntrain/batch_size)

    np.random.shuffle(idx)
    l = 0
    for bi in range(nbatch):
        ids = idx[bi*batch_size:(bi+1)*batch_size]
        sounds = Xtrain[ids, :, :]
        labels = Ytrain[ids]

        # Compute
        logprobs = model(sounds)
        loss = loss_fn(logprobs, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        l += loss.item()
    
    logprobs2 = model(Xtest)
    loss2 = loss_fn(logprobs2, Ytest)
    valid_ac = (torch.argmax(logprobs2, dim=1) == Ytest).sum()*100/NValid
    if valid_ac > valid_accuracies:
        valid_accuracies = valid_ac

    return l/nbatch, valid_accuracies

# Initialisation de la fonction de perte et de l'optimisateur
loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Boucle de l'algorithme
epochs = 100
train_losses = []
valid_accuracies = 0

start = time.time()
for t in range(epochs):
    print(f"Epoch {t+1}\n---------------")
    l_t, valid_accuracies = train_loop(model, loss_fn, optimizer, valid_accuracies)
    train_losses.append(l_t)
print("Done !")
print(f"Temps écoulé : {time.time() - start}")
print(f"Best accuracy: {valid_accuracies}")
plt.plot(train_losses)
plt.show()