# Apprentissage profond

Ce dépôt contient le code pour les travaux pratiques de deep learning.

## Installation des dépendances

Il faut d'abord installer un environnement virtuel.

`python -m venv venv`

Puis l'activer.

`source venv/bin/activate`

Ensuite l'installation des dépences se fait en fonction de la machine hôte :

- Si CUDA est disponible

`pip install -r requirements.txt`

- Sinon

`pip install -r requirements-cpu.txt`

# Exécution

Il ne faut pas oublier d'activer l'environnement virtuel.

`source venv/bin/activate`

Chaque dossier contient un fichier contennant le réseau neuronal sous forme d'une classe ainsi qu'un fichier utilisant le réseau neuronal. Il suffit d'exécuterce dernier avec python.

`python script.py`