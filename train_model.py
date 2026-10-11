# train_model.py

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from data_loader import DataLoader
from schema import ALL_COLUMNS, TARGET_COLUMN


def build_pipeline() -> Pipeline:
    """
    Construit le pipeline encodeur One-Hot + forêt aléatoire.
    """
    return Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42)),
    ])


def train_and_save_model(data_path: str, model_output_path: str, data_format: str = 'csv',
                         no_header: bool = False, cv_folds: int = 5):
    """
    Entraîne le pipeline de classification, l'évalue et le sauvegarde.
    """
    # Charger les données
    if no_header:
        loader = DataLoader(data_path=data_path, column_names=ALL_COLUMNS, data_format=data_format)
    else:
        loader = DataLoader(combined_path=data_path, data_format=data_format)
    data = loader.load_data()

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Colonne '{TARGET_COLUMN}' absente. Si le fichier n'a pas d'en-tête (fichier UCI brut), "
            "utilisez --no_header.")

    # Séparer les caractéristiques et la cible.
    # Les valeurs manquantes de 'stalk-root' ('?') sont conservées comme une catégorie à part entière.
    X = data.drop(TARGET_COLUMN, axis=1).astype(str)
    y = data[TARGET_COLUMN]

    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    model = build_pipeline()

    # Validation croisée sur le jeu d'entraînement (l'encodeur est ré-ajusté à chaque pli)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"Validation croisée ({cv_folds} plis) : exactitude {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Entraîner le modèle
    model.fit(X_train, y_train)

    # Évaluation
    y_pred = model.predict(X_test)
    print("Rapport de classification :")
    print(classification_report(y_test, y_pred))
    if len(model.classes_) == 2:
        positive_class = model.classes_[1]
        y_proba = model.predict_proba(X_test)[:, 1]
        print(f"AUC-ROC (classe positive '{positive_class}') :", roc_auc_score(y_test == positive_class, y_proba))

    # Sauvegarder le modèle
    joblib.dump(model, model_output_path)
    print("Modèle sauvegardé avec succès.")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Entraînement du Modèle de Classification des Champignons')
    parser.add_argument('--data', type=str, required=True,
                        help='Chemin vers le fichier des données Mushroom (csv, json, excel)')
    parser.add_argument('--model_output', type=str, required=True, help='Chemin pour sauvegarder le modèle entraîné')
    parser.add_argument('--data_format', type=str, default='csv',
                        help='Format du fichier des données (csv, json, excel)')
    parser.add_argument('--no_header', action='store_true',
                        help='Le fichier n\'a pas de ligne d\'en-tête (fichier UCI brut agaricus-lepiota.data)')

    args = parser.parse_args()
    train_and_save_model(
        data_path=args.data,
        model_output_path=args.model_output,
        data_format=args.data_format,
        no_header=args.no_header
    )
