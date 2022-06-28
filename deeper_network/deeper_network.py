from cProfile import label
from pickletools import optimize
import torch
from torch import nn
import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
import gzip
from torchmetrics import ConfusionMatrix
import pandas as pd
import seaborn  as sn
from sklearn.decomposition import PCA

from neuralnetwork import NeuralNetwork

# Utilisation du GPU si disponible
device = "cuda" if torch.cuda.is_available() else "cpu"

# Chargement du jeu de données
fp = gzip.open('../fashion-mnist.pk.gz', 'rb')
allXtrain, allYtrain, Xtest, Ytest, classlist  = pickle.load(fp) 

Xtrain, Ytrain  = allXtrain[:20000].to(device), allYtrain[:20000].to(device)
Xvalid, Yvalid  = allXtrain[20000:30000].to(device), allYtrain[20000:30000].to(device)
Xtest, Ytest = Xtest.to(device), Ytest.to(device)
classes = [
    'T-shirt', 'Pantalon', 'Pull', 'Robe', 'Manteau',
    'Sandales', 'Chemise', 'Baskets', 'Sac', 'Chaussures à talon'
]
color = ['r', 'g', 'b', 'c', 'purple', 'y', 'k', 'darkorange', 'saddlebrown', 'lightgreen']

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

    pred_valid = model(Xvalid)
    loss_valid = loss_fn(pred_valid, Yvalid)
    l_valid = loss_valid.item()

    return l/nbatch, l_valid


# Initialisation de la fonction de perte et de l'optimisateur
loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Boucle de l'algorithme
epochs = 100
train_losses = []
valid_losses = []

for t in range(epochs):
    print(f"Epoch {t+1}\n---------------")
    l_t, l_v = train_loop(model, loss_fn, optimizer)
    train_losses.append(l_t)
    valid_losses.append(l_v)
print("Done !")

pred_test = model(Xtest)
confmat = ConfusionMatrix(num_classes=10).to(device)
cf_matrix = confmat(pred_test, Ytest).to("cpu")
df_cm = pd.DataFrame(cf_matrix/torch.sum(cf_matrix)*10, index=classes, columns=classes)

plt.figure(figsize=(12, 7))
sn.heatmap(df_cm, annot=True)

plt.figure()
pca = PCA(n_components=2).fit_transform(Xtrain.cpu())
Ytrain = Ytrain.cpu()
for i in range(10):
    ix = np.where(Ytrain == i)
    plt.scatter(pca[ix, 0], pca[ix, 1], c=color[i], label=classes[i])
plt.xlabel("v1")
plt.ylabel("v2")
plt.legend()

plt.figure()
plt.plot(train_losses, label="Entraînement")
plt.plot(valid_losses, label="Validation")
plt.xlabel("epochs")
plt.ylabel("Pertes")
plt.legend()
plt.show()