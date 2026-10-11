# tests/test_pipeline.py

import json
import random

import pandas as pd
import pytest

import main as main_module
from data_loader import DataLoader
from features_extractor import FeatureExtractor
from model_loader import ModelLoader
from schema import ALL_COLUMNS, FEATURE_COLUMNS
from train_model import train_and_save_model


def make_rows(n=200, seed=0):
    """Jeu synthétique : l'odeur 'n' est comestible, 'p' est toxique, le reste est du bruit."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        label = rng.choice(['e', 'p'])
        row = {col: rng.choice(['a', 'b', 'c']) for col in FEATURE_COLUMNS}
        row['odor'] = 'n' if label == 'e' else 'p'
        rows.append({'class': label, **row})
    return pd.DataFrame(rows, columns=ALL_COLUMNS)


@pytest.fixture
def trained_model_path(tmp_path):
    data_path = tmp_path / 'mushroom.csv'
    make_rows().to_csv(data_path, index=False)
    model_path = tmp_path / 'model.joblib'
    train_and_save_model(str(data_path), str(model_path))
    return model_path


def write_attributes(tmp_path, **overrides):
    attributes = {col: 'a' for col in FEATURE_COLUMNS}
    attributes.update(overrides)
    path = tmp_path / 'attributes.json'
    path.write_text(json.dumps(attributes))
    return path


def test_train_then_predict(tmp_path, trained_model_path):
    edible = main_module.main(str(write_attributes(tmp_path, odor='n')), str(trained_model_path))
    poisonous = main_module.main(str(write_attributes(tmp_path, odor='p')), str(trained_model_path))

    assert edible[0] == 'e'
    assert poisonous[0] == 'p'
    assert 0.5 < edible[1] <= 1.0


def test_train_from_headerless_file(tmp_path):
    data_path = tmp_path / 'agaricus-lepiota.data'
    make_rows().to_csv(data_path, index=False, header=False)
    model_path = tmp_path / 'model.joblib'

    train_and_save_model(str(data_path), str(model_path), no_header=True)

    loader = ModelLoader(str(model_path))
    loader.load_model()
    assert loader.feature_columns() == FEATURE_COLUMNS


def test_headerless_file_without_flag_is_rejected(tmp_path):
    data_path = tmp_path / 'agaricus-lepiota.data'
    make_rows().to_csv(data_path, index=False, header=False)

    with pytest.raises(ValueError, match='--no_header'):
        train_and_save_model(str(data_path), str(tmp_path / 'model.joblib'))


def test_unseen_value_is_rejected(tmp_path, trained_model_path):
    with pytest.raises(ValueError, match="odor='z'"):
        main_module.main(str(write_attributes(tmp_path, odor='z')), str(trained_model_path))


def test_extractor_rejects_missing_and_unknown_attributes():
    extractor = FeatureExtractor(feature_columns=['odor', 'habitat'])

    with pytest.raises(ValueError, match='manquants : habitat'):
        extractor.extract_features({'odor': 'n'})
    with pytest.raises(ValueError, match='inconnus : colour'):
        extractor.extract_features({'odor': 'n', 'habitat': 'u', 'colour': 'x'})


def test_extractor_orders_columns_like_the_model():
    extractor = FeatureExtractor(feature_columns=['odor', 'habitat'])

    features = extractor.extract_features({'habitat': 'u', 'odor': 'n'})

    assert list(features.columns) == ['odor', 'habitat']
    assert features.iloc[0].tolist() == ['n', 'u']


def test_data_loader_requires_enough_parameters():
    with pytest.raises(ValueError, match='insuffisants'):
        DataLoader(data_path='data.csv').load_data()
    with pytest.raises(ValueError, match='non supporté'):
        DataLoader(combined_path='data.xml', data_format='xml').load_data()
