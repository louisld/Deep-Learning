from torch import nn


class NeuralNetwork(nn.Module):
    """
    Réseau neuronal pour apprentissage incrémental.
    """
    
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.sequence = nn.Sequential(
            nn.Linear(28*28, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        logits = self.sequence(x)
        return logits