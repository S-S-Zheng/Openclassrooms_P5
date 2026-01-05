"""
Configuration pytest et fixtures
"""

import os

# le noqa permet d'indiquer explicitement a isort d'ignorer les lignes
import pickle  # noqa: F401
import urllib.parse  # Import indispensable pour les caractères spéciaux
from pathlib import Path
from unittest.mock import create_autospec  # noqa: F401
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import yaml
from catboost import CatBoostClassifier
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.database import get_db
from app.main import app
from app.ml.model import MLModel

# ========================== MOCK/FIXTURE==========================

# =========================== DATAS ===========================


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


@pytest.fixture(params=["service_unavailable", "value_error", "unexpected_error"])
def error_responses(request):
    # On définit un dictionnaire où les clés correspondent aux params
    datas = {
        "service_unavailable": {
            "mock_args": {"is_missing": True},
            "expected_status": 503,
            "error_msg": "Modèle non chargé sur le serveur",
        },
        "value_error": {
            "mock_args": {"should_fail": True, "error_type": "value"},
            "expected_status": 422,
            "error_msg": "Modèle non chargé",
        },
        "unexpected_error": {
            "mock_args": {"should_fail": True, "error_type": "exception"},
            "expected_status": 500,
            "error_msg": "Erreur interne critique",
        },
    }
    return datas[request.param]


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


# fake csv
@pytest.fixture
def fake_csv(tmp_path):
    # faux chemin avec fichier csv
    fake_file = tmp_path / "hist_datas"
    fake_file.mkdir()
    fake_file_path = fake_file / "test_data.csv"
    fake_data = {
        "features": {
            "age": [20, 50],
            "genre": ["f", "m"],
            "revenue_mensuel": [1500, 8000],
            "a_quitte_l_entreprise": [1, 0],
        }
    }
    pd.DataFrame(fake_data["features"]).to_csv(fake_file_path, index=False)

    return fake_file_path, fake_data


# ========================= UNIT ==========================


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
def client(db_session_for_tests):
    # Injection de la session db
    app.dependency_overrides[get_db] = lambda: db_session_for_tests
    # Le "with" déclenche le lifespan (startup)
    with TestClient(app) as test_client:
        # Attentio: ce qui est sur la même ligne que yield et envoyée au test
        # ce qui est différent de ce qui est après la ligne yield qui s'exe à la fin du test
        yield test_client
    # À la sortie du "with", le lifespan (shutdown) est déclenché
    app.dependency_overrides.clear()


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

            mock.predict.return_value = (1, 0.85, "Démissionnaire")
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
                "classes": ["Employé", "Démissionnaire"],
                "threshold": 0.6,
            }

        # Injection directe dans l'application FastAPI
        app.state.model = mock
        return mock

    return _factory


# ========================= DB ==============================

# hors docker l'adresse = localhost != dans docker = db ==> os.getenv()
# ====================== Variables d'environnement ==============
# Récupération sécurisée
db_user_test = os.getenv("POSTGRES_USER_TEST")
db_pass_test = urllib.parse.quote_plus(os.getenv("POSTGRES_PASSWORD_TEST", ""))
db_host_test = os.getenv("POSTGRES_HOST_TEST")
db_port_test = os.getenv("POSTGRES_PORT_TEST")
db_name_test = os.getenv("POSTGRES_DB_TEST")

#
DATABASE_URL_TEST = f"postgresql://{db_user_test}:{db_pass_test}@{db_host_test}:{db_port_test}/{db_name_test}"  # noqa: E501
# DATABASE_URL_TEST = os.getenv("DATABASE_URL_TEST")

# ENGINE: point de départ de SQLAlchemy
test_engine = create_engine(DATABASE_URL_TEST)


@pytest.fixture
def TestingEngine():
    return test_engine


# SessionLocal est une factory à sessions pour les routes
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture
def TestingSession():
    return TestingSessionLocal


# fixture lancer automatiquement et pour la durée de la session de test
# Pour eviter de recréer les tables a chaque test les utilisant
@pytest.fixture(scope="session", autouse=True)
def init_db_for_tests():
    """Crée les tables une seule fois pour toute la session de test."""
    Base.metadata.create_all(bind=test_engine)
    yield
    # Vide la base db à la fin de la session != rollback()
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db_session_for_tests():
    """Fournit une session propre pour chaque test et nettoie après."""
    connection = test_engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    yield session

    session.close()
    # Annule l'insertion pour le test suivant plus efficace de delete
    if transaction.is_active:
        transaction.rollback()
    connection.close()


# Evite au routes d'utiliser la session normale plutot que le test à cause de Depend(get_db)
# ==> les client.post, get etc seront automatiquement envoyées vers le test.
@pytest.fixture(autouse=True)
def override_db(db_session_for_tests):
    """
    emplace get_db par la session de test pour TOUS les tests de ce fichier
    """
    app.dependency_overrides[get_db] = lambda: db_session_for_tests
    yield
    app.dependency_overrides.clear()
