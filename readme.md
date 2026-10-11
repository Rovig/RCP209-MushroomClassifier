Mushroom Classifier est une application Python modulaire permettant de classifier des champignons (comestible ou toxique) à partir de leurs attributs observés, complétés au besoin par une photo pour la couleur du chapeau, en utilisant un modèle de machine learning pré-entraîné sur le jeu de données UCI "Mushroom". Ce projet suit les principes SOLID et KISS pour assurer maintenabilité et extensibilité.

> Projet pédagogique : le résultat est indicatif et ne doit jamais servir à décider de consommer un champignon.

## Table des Matières
+ Prérequis
+ Structure du Projet
+ Modules
+ Entraînement du Modèle
+ Utilisation du Pipeline de Prédiction
+ Tests

## Prérequis
Python 3.12 ou plus récent. Installez les dépendances :

```
pip install -r requirements.txt
```

Téléchargez le jeu de données "Mushroom" depuis l'UCI Machine Learning Repository (fichier `agaricus-lepiota.data`, sans ligne d'en-tête). Un fichier CSV avec en-tête (colonne `class` puis les 22 attributs) est également accepté.

## Structure du Projet

```
mushroom_classifier/
│
├── main.py
├── schema.py
├── data_loader.py
├── image_loader.py
├── cap_color_estimator.py
├── features_extractor.py
├── model_loader.py
├── predictor.py
├── train_model.py
├── example_attributes.json
├── requirements.txt
└── tests/
```

## Modules

**main.py**
Point d'entrée du programme de prédiction.

**schema.py**
Noms des colonnes du jeu de données et libellés des classes.

**data_loader.py**
Charge les données d'entraînement (csv, json, excel), avec ou sans ligne d'en-tête.

**image_loader.py**
Charge et redimensionne une image, puis la convertit en tableau NumPy normalisé.

**cap_color_estimator.py**
Estime la couleur du chapeau (`cap-color`) : couleur dominante de la zone centrale de l'image, rapprochée de la couleur de référence la plus proche.

**features_extractor.py**
Valide les attributs fournis (attributs manquants, inconnus, valeurs jamais vues à l'entraînement) et les met au format attendu par le modèle.

**model_loader.py**
Charge le pipeline pré-entraîné à partir d'un fichier.

**predictor.py**
Prédit la classification (comestible ou toxique) et sa confiance.

**train_model.py**
Entraîne un pipeline (encodeur One-Hot + forêt aléatoire) sur le jeu de données "Mushroom", l'évalue (validation croisée et jeu de test) et le sauvegarde dans un fichier unique.

## Entraînement du Modèle
Avant d'utiliser le pipeline de prédiction, vous devez entraîner le modèle. Avec le fichier UCI brut :

```
python train_model.py --data agaricus-lepiota.data --no_header --model_output mushroom_model.joblib
```

Avec un fichier CSV comportant une ligne d'en-tête :

```
python train_model.py --data mushroom.csv --model_output mushroom_model.joblib
```

Les valeurs manquantes de `stalk-root` (`?`) sont conservées comme une catégorie à part entière.

## Utilisation du Pipeline de Prédiction
Décrivez le champignon dans un fichier JSON contenant les 22 attributs (voir `example_attributes.json`, codes définis dans `agaricus-lepiota.names`), puis :

```
python main.py --attributes example_attributes.json --model mushroom_model.joblib
```

### Avec une photo
L'option `--image` estime la couleur du chapeau à partir d'une photo lorsque `cap-color` est absent du fichier JSON ; les 21 autres attributs restent à fournir :

```
python main.py --attributes attributs.json --model mushroom_model.joblib --image champignon.jpg
```

Si `cap-color` figure déjà dans le JSON, cette valeur est conservée et l'estimation est seulement affichée. L'estimation est rudimentaire : elle suppose le chapeau cadré au centre et un éclairage neutre, et ne remplace pas une observation directe.

Ne chargez que des modèles produits par `train_model.py` : un fichier joblib peut exécuter du code arbitraire.

## Tests

```
python -m pytest
```
