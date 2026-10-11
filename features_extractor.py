# features_extractor.py

import pandas as pd
from typing import Dict, List, Optional


class FeatureExtractor:
    def __init__(self, feature_columns: List[str], known_categories: Optional[Dict[str, List[str]]] = None):
        """
        Initialise l'extracteur avec les colonnes attendues par le modèle.

        :param feature_columns: Noms des attributs attendus par le modèle, dans l'ordre d'entraînement
        :param known_categories: Valeurs vues à l'entraînement pour chaque attribut (facultatif).
                                 Si fourni, toute valeur inconnue est rejetée.
        """
        self.feature_columns = list(feature_columns)
        self.known_categories = known_categories

    def extract_features(self, attributes: Dict[str, str]) -> pd.DataFrame:
        """
        Valide les attributs observés du champignon et les met au format attendu par le modèle.

        :param attributes: Dictionnaire attribut -> code (ex: {'cap-shape': 'x', 'odor': 'n', ...})
        :return: DataFrame d'une ligne, colonnes dans l'ordre d'entraînement
        """
        missing = [col for col in self.feature_columns if col not in attributes]
        if missing:
            raise ValueError(f"Attributs manquants : {', '.join(missing)}")

        unexpected = [key for key in attributes if key not in self.feature_columns]
        if unexpected:
            raise ValueError(f"Attributs inconnus : {', '.join(unexpected)}")

        if self.known_categories is not None:
            # L'encodeur ignore les valeurs inconnues ; les refuser ici évite une prédiction
            # faite sur un attribut silencieusement mis à zéro.
            invalid = [f"{col}='{attributes[col]}'" for col in self.feature_columns
                       if attributes[col] not in self.known_categories[col]]
            if invalid:
                raise ValueError(f"Valeurs jamais vues à l'entraînement : {', '.join(invalid)}")

        return pd.DataFrame([[attributes[col] for col in self.feature_columns]], columns=self.feature_columns)
