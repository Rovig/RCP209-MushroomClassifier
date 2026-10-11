# cap_color_estimator.py

import numpy as np

# Couleurs de référence (RVB, 0-255) pour les codes 'cap-color' du jeu de données UCI.
CAP_COLORS = {
    'n': ('brun', (139, 90, 43)),
    'b': ('chamois', (240, 220, 130)),
    'c': ('cannelle', (210, 105, 30)),
    'g': ('gris', (128, 128, 128)),
    'r': ('vert', (85, 130, 60)),
    'p': ('rose', (255, 182, 193)),
    'u': ('violet', (128, 0, 128)),
    'e': ('rouge', (200, 30, 30)),
    'w': ('blanc', (245, 245, 245)),
    'y': ('jaune', (255, 215, 0)),
}


class CapColorEstimator:
    def __init__(self, levels: int = 8):
        """
        :param levels: Nombre de niveaux par canal pour regrouper les pixels de teinte proche
        """
        self.levels = levels
        self.codes = list(CAP_COLORS)
        self.references = np.array([CAP_COLORS[code][1] for code in self.codes]) / 255.0

    def dominant_color(self, image_array: np.ndarray) -> np.ndarray:
        """
        Retourne la couleur dominante (RVB, valeurs dans [0, 1]) de la zone centrale de l'image.

        Le champignon est supposé cadré au centre : seule la moitié centrale est analysée
        pour limiter l'influence de l'arrière-plan.
        """
        height, width = image_array.shape[:2]
        center = image_array[height // 4: height - height // 4, width // 4: width - width // 4]
        pixels = center.reshape(-1, 3)

        bins = np.round(pixels * (self.levels - 1)).astype(int)
        keys = (bins[:, 0] * self.levels + bins[:, 1]) * self.levels + bins[:, 2]
        dominant_key = np.bincount(keys).argmax()
        return pixels[keys == dominant_key].mean(axis=0)

    def estimate(self, image_array: np.ndarray) -> str:
        """
        Estime le code 'cap-color' : couleur de référence la plus proche de la couleur dominante.

        :param image_array: Image prétraitée par ImageLoader
        :return: Code UCI de la couleur du chapeau (ex: 'n' pour brun)
        """
        distances = np.linalg.norm(self.references - self.dominant_color(image_array), axis=1)
        return self.codes[int(distances.argmin())]
