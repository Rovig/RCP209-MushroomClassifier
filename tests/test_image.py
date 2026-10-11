# tests/test_image.py

import json

import numpy as np
import pytest
from PIL import Image

import main as main_module
from cap_color_estimator import CapColorEstimator
from image_loader import ImageLoader
from schema import FEATURE_COLUMNS
from test_pipeline import make_rows
from train_model import train_and_save_model


def save_image(path, center_rgb, background_rgb=(30, 90, 30), size=(400, 300)):
    """Image avec un rectangle central uni (le champignon) sur un fond uni."""
    image = Image.new('RGB', size, background_rgb)
    width, height = size
    image.paste(center_rgb, (width // 5, height // 5, width - width // 5, height - height // 5))
    image.save(path)
    return path


def test_loader_resizes_and_normalises(tmp_path):
    path = save_image(tmp_path / 'mushroom.png', (255, 255, 255))

    image_array = ImageLoader(str(path)).load_and_preprocess()

    assert image_array.shape == (224, 224, 3)
    assert image_array.min() >= 0.0 and image_array.max() == 1.0


def test_loader_rejects_missing_and_invalid_files(tmp_path):
    not_an_image = tmp_path / 'notes.png'
    not_an_image.write_text('pas une image')

    with pytest.raises(ValueError, match="chargement de l'image"):
        ImageLoader(str(tmp_path / 'absent.png')).load_and_preprocess()
    with pytest.raises(ValueError, match="chargement de l'image"):
        ImageLoader(str(not_an_image)).load_and_preprocess()


@pytest.mark.parametrize('rgb, expected', [
    ((120, 80, 40), 'n'),
    ((250, 250, 250), 'w'),
    ((210, 25, 25), 'e'),
    ((250, 210, 20), 'y'),
    ((125, 125, 125), 'g'),
])
def test_estimator_reads_the_central_colour(tmp_path, rgb, expected):
    path = save_image(tmp_path / 'mushroom.png', rgb)
    image_array = ImageLoader(str(path)).load_and_preprocess()

    assert CapColorEstimator().estimate(image_array) == expected


def test_estimator_ignores_the_background():
    image_array = np.zeros((224, 224, 3))
    image_array[:, :] = (0.1, 0.4, 0.1)
    image_array[56:168, 56:168] = (0.96, 0.96, 0.96)

    assert CapColorEstimator().estimate(image_array) == 'w'


@pytest.fixture
def trained_model_path(tmp_path):
    # Les valeurs de cap-color du jeu synthétique sont remplacées par de vrais codes de couleur.
    data = make_rows()
    data['cap-color'] = ['n' if i % 2 else 'w' for i in range(len(data))]
    data_path = tmp_path / 'mushroom.csv'
    data.to_csv(data_path, index=False)
    model_path = tmp_path / 'model.joblib'
    train_and_save_model(str(data_path), str(model_path))
    return model_path


def write_attributes(tmp_path, without=(), **overrides):
    attributes = {col: 'a' for col in FEATURE_COLUMNS}
    attributes.update(overrides)
    for col in without:
        del attributes[col]
    path = tmp_path / 'attributes.json'
    path.write_text(json.dumps(attributes))
    return path


def test_image_supplies_the_missing_cap_colour(tmp_path, trained_model_path, capsys):
    attributes = write_attributes(tmp_path, without=['cap-color'], odor='n')
    image = save_image(tmp_path / 'mushroom.png', (120, 80, 40))

    prediction, _ = main_module.main(str(attributes), str(trained_model_path), str(image))

    assert prediction == 'e'
    assert 'brun (n)' in capsys.readouterr().out


def test_cap_colour_is_required_without_image(tmp_path, trained_model_path):
    attributes = write_attributes(tmp_path, without=['cap-color'], odor='n')

    with pytest.raises(ValueError, match='manquants : cap-color'):
        main_module.main(str(attributes), str(trained_model_path))


def test_explicit_cap_colour_wins_over_the_image(tmp_path, trained_model_path, capsys):
    attributes = write_attributes(tmp_path, odor='n', **{'cap-color': 'w'})
    image = save_image(tmp_path / 'mushroom.png', (120, 80, 40))

    main_module.main(str(attributes), str(trained_model_path), str(image))

    assert "l'estimation est ignorée" in capsys.readouterr().out
