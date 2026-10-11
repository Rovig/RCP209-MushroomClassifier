class Predictor:
    def __init__(self, model):
        self.model = model

    def predict(self, features):
        """
        Prend une ligne de caractéristiques et retourne la classe prédite ainsi qu'une note de confiance.
        """
        probabilities = self.model.predict_proba(features)[0]
        best = probabilities.argmax()
        return self.model.classes_[best], float(probabilities[best])
