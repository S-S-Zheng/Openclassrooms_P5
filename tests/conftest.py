"""
Configuration pytest et fixtures
"""

# le noqa permet d'indiquer explicitement a isort d'ignorer les lignes
import pickle  # noqa: F401
from unittest.mock import create_autospec  # noqa: F401
from unittest.mock import MagicMock

import numpy as np
import pytest
from catboost import CatBoostClassifier
from fastapi.testclient import TestClient

from app.main import app

# ========================== MOCK/FIXTURE==========================


# dataset de test de 5 observations x 3 features
@pytest.fixture
def sample_features():
    return np.random.randn(5, 3)


# pickle.load
@pytest.fixture
def mock_pickle(monkeypatch):
    fake_data = {"features_names": ["f1", "f2"], "thresh": 0.6}

    def fake_load(file):
        filename = str(file)
        for key, value in fake_data.items():
            if key in filename:
                return value
        return None

    monkeypatch.setattr("pickle.load", fake_load)


# catboost
@pytest.fixture
def mock_catboost(monkeypatch):
    # create_autospec crée mock exact mêmes méthodes que vrai CatBoost mais
    # Problème avec le nom donc passage a MagicMock(spec=)
    mock_model = MagicMock(spec=CatBoostClassifier)
    # Besoin de tricher afin que le type du mock oit bien "CatBoostClassifier"
    type(mock_model).__name__ = "CatBoostClassifier"
    # Par défaut MagicMock ne fait rien mais on écrit None pour se souvenir
    # qu'il y a cette fonction équivalent à pass ici
    mock_model.load_model.return_value = None
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
    mock_model.predict.return_value = np.array([1])
    mock_model.get_feature_importance.return_value = np.array([0.2, 0.5, 0.3])

    monkeypatch.setattr("app.ml.model.CatBoostClassifier", lambda: mock_model)
    return mock_model


# Client FastAPI
@pytest.fixture
def client():
    return TestClient(app)


# MLModel
@pytest.fixture
def mock_ml_model(monkeypatch):
    def _factory(should_fail=True, error_type="value", endpoint="predict"):
        mock = MagicMock()
        if should_fail:
            if error_type == "value":
                mock.predict.side_effect = ValueError("Modèle non chargé")
                mock.get_feature_importance.side_effect = ValueError(
                    "Modèle non chargé"
                )
                mock.get_model_info.side_effect = ValueError("Modèle non chargé")
            else:
                mock.predict.side_effect = Exception("Erreur interne critique")
                mock.get_feature_importance.side_effect = Exception(
                    "Erreur interne critique"
                )
                mock.get_model_info.side_effect = Exception("Erreur interne critique")
        else:
            mock.predict.return_value = (1.0, 0.85, "Démissionaire")
            mock.get_feature_importance.return_value = [
                ("f1", 0.9),
                ("f2", 0.6),
                ("f3", 0.55),
                ("f4", 0.53),
                ("f5", 0.5),
            ]
            mock.get_model_info.return_value = {
                "model_type": "CatBoostClassifier",
                "n_features": len(["f1", "f2", "f3", "f4", "f5"]),
                "classes": ["Employé", "Démissionaire"],
                "threshold": 0.6,
            }

        monkeypatch.setattr(f"app.api.routes.{endpoint}.ml_model", mock)
        return mock

    return _factory
