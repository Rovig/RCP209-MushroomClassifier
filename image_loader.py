from PIL import Image, UnidentifiedImageError
import numpy as np

class ImageLoader:
    def __init__(self, image_path, size=(224, 224)):
        """
        Initialise le loader avec le chemin de l'image et la taille de redimensionnement.

        :param image_path: Chemin vers l'image du champignon
        :param size: Tuple pour redimensionner l'image
        """
        self.image_path = image_path
        self.size = size

    def load_and_preprocess(self):
        """
        Charge l'image, la redimensionne et la convertit en tableau NumPy normalisé.

        :return: Tableau NumPy de l'image prétraitée, de forme (hauteur, largeur, 3), valeurs dans [0, 1]
        """
        try:
            with Image.open(self.image_path) as image:
                image = image.convert('RGB').resize(self.size)
                return np.array(image) / 255.0  # Normalisation
        except (OSError, UnidentifiedImageError) as e:
            raise ValueError(f"Erreur lors du chargement de l'image: {e}") from e
