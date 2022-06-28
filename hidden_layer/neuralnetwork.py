from torch import nn


class NeuralNetwork(nn.Module):
    """
    Réseau neuronal pour apprentissage incrémental
    avec une couche cachée.
    """
    
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.sequence = nn.Sequential(
            nn.Linear(28*28, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        logits = self.sequence(x)
        return logits