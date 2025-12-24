"""
Configuration pytest et fixtures
"""

# le noqa permet d'indiquer explicitement a isort d'ignorer les lignes
import pickle  # noqa: F401
from unittest.mock import MagicMock, create_autospec

import numpy as np
import pytest
import shap  # noqa: F401
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
    # create_autospec crée mock exact mêmes méthodes que vrai CatBoost
    mock_model = create_autospec(CatBoostClassifier, instance=True)
    # Par défaut MagicMock ne fait rien mais on écrit None pour se souvenir
    # qu'il y a cette fonction équivalent à pass ici
    mock_model.load_model.return_value = None
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
    mock_model.predict.return_value = np.array([1])
    mock_model.get_feature_importance.return_value = np.array([0.2, 0.5, 0.3])

    monkeypatch.setattr("app.ml.model.CatBoostClassifier", lambda: mock_model)
    return mock_model


# shap
# create_autospec excellent pour les classes type CBC mais fragile pour les
# bibliothèques type shap (structure shap dynamique) ==> MagicMock plutôt
@pytest.fixture
def mock_shap(monkeypatch):
    # 1. Mock de l'objet Explanation (ce que renvoie l'explainer)
    mock_explanation = MagicMock()
    # Pas de return_values car on consulte les attributs values, base_values,
    # data de TreeExplainer
    mock_explanation.values = np.random.randn(5, 3)
    mock_explanation.base_values = np.zeros(5)
    mock_explanation.data = np.random.randn(5, 3)
    # Important : simuler le slicing explanation[i]
    # __getitem__ doit retourner l'objet lui-même ou une version réduite
    mock_explanation.__getitem__.return_value = mock_explanation

    # 2. Mock du TreeExplainer
    mock_explainer = MagicMock()
    # return_value ici car on appelle la fonction TreeExplainer
    mock_explainer.return_value = mock_explanation  # Pour l'appel __call__

    # 3. Patch des fonctions de visualisation
    # En les mockant ==> evite l'ouverture auto des fenetres pour rien
    # on prefixe par _ pour indiquer que la var est volontaire inutilisée
    monkeypatch.setattr("shap.summary_plot", MagicMock())
    monkeypatch.setattr("shap.plots.scatter", MagicMock())
    monkeypatch.setattr("shap.plots.waterfall", MagicMock())

    monkeypatch.setattr("shap.TreeExplainer", lambda model: mock_explainer)

    return mock_explainer


# Client FastAPI
@pytest.fixture
def client():
    return TestClient(app)


# MLModel
@pytest.fixture
def mock_ml_model(monkeypatch):
    def _factory(should_fail=True, error_type="value"):
        mock = MagicMock()
        if should_fail:
            if error_type == "value":
                mock.predict.side_effect = ValueError("Modèle non chargé")
            else:
                mock.predict.side_effect = Exception("Erreur interne critique")
        else:
            mock.predict.return_value = (1.0, 0.85, "Démissionaire")

        monkeypatch.setattr("app.api.routes.predict.ml_model", mock)
        return mock

    return _factory
