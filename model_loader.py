import joblib


class ModelLoader:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        """
        Charge le pipeline (encodeur + classifieur) à partir du chemin spécifié.

        joblib exécute le contenu du fichier : ne charger que des modèles produits par train_model.py.
        """
        self.model = joblib.load(self.model_path)
        return self.model

    def feature_columns(self):
        """
        Retourne les noms des attributs attendus par le modèle chargé.
        """
        return list(self.model.feature_names_in_)

    def known_categories(self):
        """
        Retourne, pour chaque attribut, les valeurs vues à l'entraînement.
        """
        encoder = self.model.named_steps['encoder']
        return {col: list(categories) for col, categories in zip(self.feature_columns(), encoder.categories_)}
