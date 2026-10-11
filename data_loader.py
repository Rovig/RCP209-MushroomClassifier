# data_loader.py

import pandas as pd
import json
from typing import List, Optional


class DataLoader:
    def __init__(self, names_path: Optional[str] = None, data_path: Optional[str] = None,
                 combined_path: Optional[str] = None, data_format: str = 'csv',
                 column_names: Optional[List[str]] = None):
        """
        DataLoader peut charger des données depuis des fichiers combinés ou séparés.

        :param names_path: Chemin vers le fichier contenant les noms des colonnes (facultatif)
        :param data_path: Chemin vers le fichier contenant les données (facultatif)
        :param combined_path: Chemin vers le fichier combiné contenant à la fois les noms et les données
        :param data_format: Format du fichier combiné ou des données ('csv', 'json', 'excel')
        :param column_names: Noms des colonnes pour un data_path sans en-tête (alternative à names_path)
        """
        self.names_path = names_path
        self.data_path = data_path
        self.combined_path = combined_path
        self.data_format = data_format
        self.column_names = column_names

    def load_data(self) -> pd.DataFrame:
        """
        Charge les données dans un DataFrame pandas.

        :return: DataFrame contenant les données chargées
        """
        if self.data_format not in ('csv', 'json', 'excel'):
            raise ValueError(f"Format de données non supporté : {self.data_format}")

        if self.combined_path:
            if self.data_format == 'csv':
                return pd.read_csv(self.combined_path)
            if self.data_format == 'json':
                return pd.read_json(self.combined_path)
            return pd.read_excel(self.combined_path)

        if self.data_path and (self.names_path or self.column_names):
            column_names = self._load_column_names()
            if self.data_format == 'csv':
                return pd.read_csv(self.data_path, names=column_names)
            if self.data_format == 'json':
                # Supposer que data_path est un tableau JSON avec des données correspondant aux noms de colonnes
                with open(self.data_path, 'r') as f:
                    data = json.load(f)
                return pd.DataFrame(data, columns=column_names)
            return pd.read_excel(self.data_path, names=column_names, header=None)

        raise ValueError(
            "Paramètres insuffisants pour charger les données. Fournissez soit combined_path, "
            "soit data_path avec names_path ou column_names.")

    def _load_column_names(self) -> List[str]:
        if self.column_names:
            return list(self.column_names)
        with open(self.names_path, 'r') as f:
            return [name.strip() for name in f if name.strip()]
