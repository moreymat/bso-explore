# bso-explore
Exploration des données du Baromètre de la Science Ouverte


## Installation

On crée un environment conda avec les bibliothèques habituelles pour l'analyse et la visualisation de données, plus UMAP :
```sh
conda create -n bso python=3.7 pandas scikit-learn matplotlib seaborn datashader
conda activate bso
conda install -c conda-forge umap-learn
```

### TensorFlow

L'installation de TensorFlow peut se révéler un peu plus compliquée.
Nous avons besoin de **TensorFlow>=2.0.0**, son hub (pour les modèles pré-entraînés) et ses utilitaires spécialisés pour les données textuelles:

* via conda
(préférable mais actuellement cassé https://github.com/tensorflow/text/issues/192)

```sh
conda install tensorflow==2.0.0 tensorflow-hub
# or ? conda install tensorflow-mkl==2.0.0 tensorflow-hub
pip install -U tensorflow-text
```

* via pip

```sh
pip install -U tensorflow tensorflow-hub tensorflow-text
```

## Utilisation

TODO

