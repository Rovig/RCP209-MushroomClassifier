# main.py

import argparse
import json
from features_extractor import FeatureExtractor
from model_loader import ModelLoader
from predictor import Predictor
from schema import CLASS_LABELS


def main(attributes_path, model_path):
    # Étape 1 : Charger les attributs observés du champignon
    with open(attributes_path, 'r') as f:
        attributes = json.load(f)

    # Étape 2 : Charger le modèle pré-entrainé
    model_loader = ModelLoader(model_path=model_path)
    model = model_loader.load_model()

    # Étape 3 : Valider et mettre en forme les attributs
    extractor = FeatureExtractor(feature_columns=model_loader.feature_columns(),
                                 known_categories=model_loader.known_categories())
    features = extractor.extract_features(attributes)

    # Étape 4 : Faire la prédiction
    predictor = Predictor(model=model)
    prediction, confidence = predictor.predict(features)

    # Étape 5 : Interpréter les résultats
    if prediction not in CLASS_LABELS:
        raise ValueError(f"Classe prédite inattendue : {prediction}")
    classification = CLASS_LABELS[prediction]

    print(f"Champignon est {classification} avec une confiance de {confidence * 100:.2f}%")
    print("Résultat indicatif issu d'un projet pédagogique : ne jamais s'y fier pour consommer un champignon.")
    return prediction, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classification des Champignons')
    parser.add_argument('--attributes', type=str, required=True,
                        help='Chemin vers le fichier JSON des attributs du champignon')
    parser.add_argument('--model', type=str, required=True, help='Chemin vers le modèle pré-entrainé')

    args = parser.parse_args()
    main(args.attributes, args.model)
