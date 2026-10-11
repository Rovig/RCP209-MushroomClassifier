# main.py

import argparse
import json
from cap_color_estimator import CAP_COLORS, CapColorEstimator
from features_extractor import FeatureExtractor
from image_loader import ImageLoader
from model_loader import ModelLoader
from predictor import Predictor
from schema import CLASS_LABELS

CAP_COLOR = 'cap-color'


def main(attributes_path, model_path, image_path=None):
    # Étape 1 : Charger les attributs observés du champignon
    with open(attributes_path, 'r') as f:
        attributes = json.load(f)

    # Étape 2 : Estimer la couleur du chapeau à partir de l'image, si elle n'est pas déjà renseignée
    if image_path:
        image_array = ImageLoader(image_path=image_path).load_and_preprocess()
        estimated = CapColorEstimator().estimate(image_array)
        print(f"Couleur du chapeau estimée d'après l'image : {CAP_COLORS[estimated][0]} ({estimated})")
        if CAP_COLOR in attributes:
            print(f"'{CAP_COLOR}' est déjà renseigné ({attributes[CAP_COLOR]}) : l'estimation est ignorée.")
        else:
            attributes[CAP_COLOR] = estimated

    # Étape 3 : Charger le modèle pré-entrainé
    model_loader = ModelLoader(model_path=model_path)
    model = model_loader.load_model()

    # Étape 4 : Valider et mettre en forme les attributs
    extractor = FeatureExtractor(feature_columns=model_loader.feature_columns(),
                                 known_categories=model_loader.known_categories())
    features = extractor.extract_features(attributes)

    # Étape 5 : Faire la prédiction
    predictor = Predictor(model=model)
    prediction, confidence = predictor.predict(features)

    # Étape 6 : Interpréter les résultats
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
    parser.add_argument('--image', type=str,
                        help='Chemin vers l\'image du champignon, pour estimer la couleur du chapeau '
                             'si \'cap-color\' est absent des attributs')

    args = parser.parse_args()
    main(args.attributes, args.model, args.image)
