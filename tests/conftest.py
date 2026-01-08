"""
Module de configuration globale et de définition des fixtures pour Pytest.

Ce fichier est le pilier de la suite de tests. Il centralise la configuration
des environnements de test, notamment :
1. La gestion d'une base de données PostgreSQL de test isolée.
2. Le mockage du modèle CatBoost et des dépendances système (Pickle, OS).
3. L'injection de dépendances pour le client de test FastAPI.
4. La génération de jeux de données fictifs (profils YAML, fichiers CSV temporaires).

L'utilisation de fixtures permet de garantir l'indépendance des tests et la
reproductibilité des scénarios de succès et d'échec.
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
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.database import get_db
from app.db.models_db import PredictionRecord, RequestLog  # noqa: F401
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
    """
    Fournit un dictionnaire complet de caractéristiques valides.
    Utilisé pour tester les payloads de l'endpoint /predict.
    """
    return {
        "features": {
            "age": 32,
            "genre": "m",
            "revenu_mensuel": 60000,
            "statut_marital": "marié(e)",
            "poste": "cadre commercial",
            "annees_dans_le_poste_actuel": 4,
            "heure_supplementaires": "oui",
            "augementation_salaire_precedente": 3,
            "nombre_participation_pee": 6,
            "nb_formations_suivies": 3,
            "distance_domicile_travail": 10,
            "niveau_education": 3,
            "domaine_etude": "infra & cloud",
            "frequence_deplacement": "frequent",
            "evolution_note": 1,
            "stagnation_promo": 0.0,
            "freq_chgt_poste": 0.1,
            "revenu_mensuel_ajuste_par_nv_hierarchique": 60000.0,
            "revenu_mensuel_par_annee_xp": 6658.88889,
            "freq_chgt_responsable": 0.1,
            "satisfaction_globale_employee": 10,
        }
    }


@pytest.fixture(params=["service_unavailable", "value_error", "unexpected_error"])
def error_responses(request):
    """
    On définit un dictionnaire où les clés correspondent aux params
    """
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
    """
    Charge dynamiquement un profil YAML selon le paramètre fourni.
    """
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
    """
    Crée un fichier CSV temporaire simulant un dataset historique.
    Utilisé pour tester les scripts d'importation.
    """
    # faux chemin avec fichier csv
    fake_file = tmp_path / "hist_datas"
    fake_file.mkdir()
    fake_file_path = fake_file / "test_data.csv"
    fake_data = {
        "features": {
            "age": [20, 50],
            "genre": ["f", "m"],
            "revenu_mensuel": [1500, 8000],
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


# Utilisé pour tests unitaire en subsituant CBC
@pytest.fixture
def mock_catboost(monkeypatch):
    """
    Mock de bas niveau ciblant la bibliothèque CatBoostClassifier.

    Cette fixture utilise 'monkeypatch' pour intercepter l'instanciation de
    CatBoostClassifier au sein du module 'app.ml.model'. Elle permet de tester
    la logique de la classe 'MLModel' sans nécessiter de fichiers binaires
    réels ni de calculs matriciels lourds.

    Elle simule les attributs et méthodes critiques :
    - .feature_names_ : pour la validation des colonnes.
    - .predict_proba() : pour le calcul des scores de confiance.
    - .get_feature_importance() : pour l'explicabilité.

    Returns:
        MagicMock: Une instance simulant un objet CatBoostClassifier pré-entraîné.
    """
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
    """
    Fournit un TestClient FastAPI configuré avec une session de base de données de test.
    Gère le cycle de vie (lifespan) de l'application pour chaque test.
    """
    """
    Fournit un client HTTP configuré pour tester les endpoints de l'API.

    Cette fixture utilise le 'TestClient' de FastAPI au sein d'un gestionnaire
    de contexte afin de déclencher les événements 'lifespan' (startup/shutdown).
    Cela permet notamment au modèle ML d'être chargé en mémoire avant l'exécution
    des tests.

    Note:
        L'injection de la base de données n'est plus gérée ici car elle est
        assurée globalement par la fixture 'override_db'.

    Yields:
        TestClient: Une instance de client capable d'effectuer des requêtes (GET, POST, etc.).
    """
    with TestClient(app) as test_client:
        yield test_client


# MLModel pour tester la relation API/ML
@pytest.fixture
def mock_ml_model(func_sample):
    """
    Factory de mocks pour simuler différents états du modèle ML au sein de l'application.

    Cette fixture retourne une fonction usine permettant de configurer dynamiquement
    le comportement du modèle (succès, erreur, ou absence) et de l'injecter
    directement dans l'état de l'application FastAPI (`app.state.model`).

    Args:
        func_sample (dict): Fixture fournissant un exemple de features pour
            calculer les métadonnées de réponse.

    Returns:
        Callable: Une fonction `_factory` acceptant les paramètres suivants :
            - is_missing (bool): Si True, simule l'absence totale de modèle chargé.
            - should_fail (bool): Si True, les méthodes du modèle lèveront une exception.
            - error_type (str): Type d'erreur à lever ('value' pour ValueError,
                sinon Exception générique).

    Note:
        Le mock généré simule l'intégralité de l'interface `MLModel` :
        - `predict()` : Retourne un tuple (classe, confiance, nom).
        - `get_feature_importance()` : Retourne une liste de tuples (feature, importance).
        - `get_model_info()` : Retourne un dictionnaire de métadonnées.
    """

    def _factory(is_missing=False, should_fail=True, error_type="value"):
        mock = MagicMock()
        # Cas 1 : Simulation du modèle non initialisé (Lifespan défaillant)
        if is_missing:
            app.state.model = None
            return None
        # Cas 2 : Simulation de pannes durant l'inférence ou l'analyse
        if should_fail:
            error = (
                ValueError("Modèle non chargé")
                if error_type == "value"
                else Exception("Erreur interne critique")
            )
            mock.predict.side_effect = error
            mock.get_feature_importance.side_effect = error
            mock.get_model_info.side_effect = error
        # Cas 3 : Simulation d'un fonctionnement nominal (Happy Path)
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
DATABASE_URL_TEST = (
    f"postgresql+psycopg2://{db_user_test}:{db_pass_test}@{db_host_test}:"
    f"{db_port_test}/{db_name_test}"
)
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
    """
    Fixture de session : Crée les tables au démarrage et les supprime à la fin de la session.
    Utilisation automatique de la fixture.
    """
    Base.metadata.create_all(bind=test_engine)
    yield
    # Vide la base db à la fin de la session != rollback()
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db_session_for_tests():
    """
    Fournit une session de base de données isolée pour chaque test.
    Utilise une transaction SQL pour effectuer un rollback systématique à la fin
    du test, garantissant une base propre pour le test suivant.
    """
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
    Assure l'isolation systématique de la base de données pour l'application.

    En utilisant 'autouse=True', cette fixture garantit que n'importe quelle route
    faisant appel à la dépendance 'get_db' recevra la session de test en cours,
    évitant ainsi toute écriture accidentelle dans la base de production ou de dev.

    Args:
        db_session_for_tests: La session SQLAlchemy transactionnelle définie plus haut.
    """
    app.dependency_overrides[get_db] = lambda: db_session_for_tests
    yield
    app.dependency_overrides.clear()


# Session db qui saute pendant une transaction
@pytest.fixture
def db_session_broken_for_tests(db_session_for_tests):
    """
    Simule une panne de base de données (OperationalError).
    Utilisé pour tester la robustesse des rollbacks et la gestion des erreurs API.
    """
    # Le flush() entrainera un crash
    db_session_for_tests.flush = MagicMock(
        side_effect=OperationalError("Unexpected Crash", params=None, orig=None)
    )
    # On "espionne" le rollback
    db_session_for_tests.rollback = MagicMock(wraps=db_session_for_tests.rollback)
    return db_session_for_tests
