"""
Configuration pytest et fixtures
"""

import numpy as np
import pytest


# ====================== MOCKS  =======================
# Version mock de CBC
class DummyCatBoost:
    def load_model(self, path):
        pass

    def predict_proba(self, X):
        return np.array([[0.3, 0.7]])

    def predict(self, X):
        return np.array([1])

    def get_feature_importance(self):
        return np.array([0.2, 0.5, 0.3])


# Version mock de shap.Explanation
class DummyShapExplanation:
    def __init__(self, n_samples=5, n_features=3):
        self.values = np.random.randn(n_samples, n_features)
        self.base_values = np.zeros(n_samples)
        self.data = np.random.randn(n_samples, n_features)
        self.feature_names = [f"f{i+1}" for i in range(n_features)]

    def __getitem__(self, key):
        return self


# mock du TreeExplainer
class DummyShapExplainer:
    def __call__(self, X):
        n_samples, n_features = X.shape
        return DummyShapExplanation(n_samples, n_features)


# ========================== FIXTURES ==========================
# pickle.load
@pytest.fixture
def mock_pickle(monkeypatch):
    def fake_load(file):
        if "features_names" in str(file.name):
            return ["f1", "f2"]
        if "thresh" in str(file.name):
            return 0.6
        return None

    monkeypatch.setattr("pickle.load", fake_load)


# catboost
@pytest.fixture
def mock_catboost(monkeypatch):
    monkeypatch.setattr("app.ml.model.CatBoostClassifier", lambda: DummyCatBoost())


# shap
@pytest.fixture
def mock_shap(monkeypatch):
    monkeypatch.setattr(
        "app.ml.model.shap.TreeExplainer", lambda model: DummyShapExplainer()
    )
