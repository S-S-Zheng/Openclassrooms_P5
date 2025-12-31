"""
Configuration pytest et fixtures
"""

# le noqa permet d'indiquer explicitement a isort d'ignorer les lignes
import pickle  # noqa: F401
from pathlib import Path
from unittest.mock import create_autospec  # noqa: F401
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml
from catboost import CatBoostClassifier
from fastapi.testclient import TestClient

from app.main import app
from app.ml.model import MLModel

# ========================== MOCK/FIXTURE==========================


# dict de test
@pytest.fixture
def fake_dict():
    return {"f1": "1.0", "f2": "2.0", "f3": 3.0, "f4": 4.0, "f5": 5.0}


# dict de test partie fonctionnelle
@pytest.fixture
def func_sample():
    return {
        "features": {
            "genre": "m",
            "statut_marital": "célibataire",
            "age": 30,
            "revenu_mensuel": 2500,
            "augementation_salaire_precedente": 10.0,
        }
    }


# pickle.load (Fallback)
@pytest.fixture
def mock_pickle(monkeypatch):
    fake_data = {
        "feature_names": ["f1", "f2", "f3", "f4", "f5"],
        "thresh": np.array([0.6]),
    }

    def fake_load(file):
        filename = str(file)
        for key, value in fake_data.items():
            if key in filename:
                return value
        return None

    monkeypatch.setattr("pickle.load", fake_load)


# instance MLModel pour test
@pytest.fixture
def ml_model():
    return MLModel()


# catboost
@pytest.fixture
def mock_catboost(monkeypatch):
    # create_autospec crée mock exact mêmes méthodes que vrai CatBoost mais
    # Problème avec le nom donc passage a MagicMock(spec=)
    mock_model = MagicMock(spec=CatBoostClassifier)
    # Besoin de tricher afin que le type du mock soit bien "CatBoostClassifier"
    type(mock_model).__name__ = "CatBoostClassifier"
    # Ajout des attributs sur les features
    mock_model.feature_names_ = ["f1", "f2", "f3", "f4", "f5"]
    mock_model.get_cat_feature_indices.return_value = [0, 1]
    # Par défaut MagicMock ne fait rien mais on écrit None pour se souvenir
    # qu'il y a cette fonction équivalent à pass ici
    mock_model.load_model.return_value = None
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
    mock_model.predict.return_value = np.array([1])
    mock_model.get_feature_importance.return_value = np.array(
        [0.9, 0.6, 0.55, 0.53, 0.5]
    )

    monkeypatch.setattr("app.ml.model.CatBoostClassifier", lambda: mock_model)
    return mock_model


# =========================== API ===========================


# Client FastAPI
@pytest.fixture
def client():
    # Le "with" déclenche le lifespan (startup)
    with TestClient(app) as c:
        yield c
    # À la sortie du "with", le lifespan (shutdown) est déclenché


# MLModel
@pytest.fixture
def mock_ml_model(func_sample):
    def _factory(is_missing=False, should_fail=True, error_type="value"):
        mock = MagicMock()
        if is_missing:
            app.state.model = None
            return None
        if should_fail:
            error = (
                ValueError("Modèle non chargé")
                if error_type == "value"
                else Exception("Erreur interne critique")
            )
            mock.predict.side_effect = error
            mock.get_feature_importance.side_effect = error
            mock.get_model_info.side_effect = error
        else:
            features = list(func_sample["features"].keys())
            shap = [0.9, 0.6, 0.55, 0.53, 0.5]
            importance = list(zip(features, shap))

            mock.predict.return_value = (1.0, 0.85, "Démissionaire")
            mock.get_feature_importance.return_value = sorted(
                importance, key=lambda x: x[1], reverse=True
            )
            mock.feature_names_ = features
            mock.get_model_info.return_value = {
                "model_type": "CatBoostClassifier",
                "n_features": len(features),
                "feature_names": features,
                "cat_features": ["genre", "statut_marital"],
                "num_features": [
                    "age",
                    "revenu_mensuel",
                    "augementation_salaire_precedente",
                ],
                "classes": ["Employé", "Démissionaire"],
                "threshold": 0.6,
            }

        # Injection directe dans l'application FastAPI
        app.state.model = mock
        return mock

    return _factory


# Charge des profils fonctionnels
@pytest.fixture
def functionnal_profile(request):
    """Charge dynamiquement un profil YAML selon le paramètre fourni."""
    # 'request.param' contiendra le nom du profil (ex: 'happy_path')
    profile_name = request.param
    file_path = Path(__file__).parent / "fixtures" / f"fake_profile_{profile_name}.yml"

    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        data["_profile_name"] = profile_name
        return data
