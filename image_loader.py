from PIL import Image
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

        :return: Tableau NumPy de l'image prétraitée
        """
        try:
            image = Image.open(self.image_path).convert('RGB')
            image = image.resize(self.size)
            image_array = np.array(image) / 255.0  # Normalisation
            return image_array
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement de l'image: {e}")