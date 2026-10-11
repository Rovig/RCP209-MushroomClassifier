# schema.py

# Colonnes du jeu de données UCI "Mushroom" (agaricus-lepiota.data), dans l'ordre du fichier brut.
TARGET_COLUMN = 'class'

FEATURE_COLUMNS = [
    'cap-shape',
    'cap-surface',
    'cap-color',
    'bruises',
    'odor',
    'gill-attachment',
    'gill-spacing',
    'gill-size',
    'gill-color',
    'stalk-shape',
    'stalk-root',
    'stalk-surface-above-ring',
    'stalk-surface-below-ring',
    'stalk-color-above-ring',
    'stalk-color-below-ring',
    'veil-type',
    'veil-color',
    'ring-number',
    'ring-type',
    'spore-print-color',
    'population',
    'habitat',
]

ALL_COLUMNS = [TARGET_COLUMN] + FEATURE_COLUMNS

CLASS_LABELS = {'e': 'Comestible', 'p': 'Toxique'}
