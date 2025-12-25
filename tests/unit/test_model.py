"""
Tests du modèle ML
"""

import numpy as np  # noqa: F401
import pytest

from app.ml.model import MLModel

# =============================== INIT =======================================


# test methode init() ()
def test_init():
    ml = MLModel()

    assert ml.model is None
    assert ml.features_names is None
    assert ml.threshold is None
    assert ml.classes == ["Employé", "Démissionaire"]


# ======================== GET MODEL INFO =================================


# Happy path + threshold limite
@pytest.mark.parametrize("test_threshold", [0.6, None])
def test_get_model_info(mock_catboost, test_threshold):
    ml = MLModel()

    ml.model = mock_catboost
    ml.features_names = ["f1", "f2", "f3", "f4", "f5"]
    ml.classes = ["Employé", "Démissionaire"]
    ml.threshold = test_threshold

    model_info = ml.get_model_info()

    assert isinstance(model_info, dict)
    # Rq: create_autospec(catboost.CatBoostClassifier) implique que
    # le nom du type doit être celui de la classe mockée
    assert model_info["model_type"] == "CatBoostClassifier"
    assert model_info["n_features"] == 5
    assert model_info["classes"] == ["Employé", "Démissionaire"]
    # Avec le décorateur on test les deux valeurs
    assert model_info["threshold"] == test_threshold


# Echec de chargement du modele
def test_get_model_info_not_loaded():
    ml = MLModel()

    with pytest.raises(ValueError):
        ml.get_model_info()


# ================================= LOAD ======================================


# test de la méthode load() (model, features, seuil)
# load - CAS nominal
def test_load_model_success(tmp_path, mock_pickle, mock_catboost):

    model_file = tmp_path / "model.cbm"
    features_file = tmp_path / "features_names.pkl"
    threshold_file = tmp_path / "thresh_opt.pkl"

    model_file.touch()
    features_file.touch()
    threshold_file.touch()

    # test les chemins ici via des fichiers temporaires
    ml = MLModel(
        model_path=model_file,
        features_names_path=features_file,
        threshold_path=threshold_file,
    )

    ml.load()

    assert ml.model is not None
    assert ml.features_names == ["f1", "f2"]
    assert ml.threshold == 0.6


# =======================================================================


# load - CAS model manquant
def test_load_model_failed(
    tmp_path,
):
    model_file = tmp_path / "model.cbm"
    features_file = tmp_path / "features_names.pkl"
    threshold_file = tmp_path / "thresh_opt.pkl"
    ml = MLModel(
        model_path=model_file,
        features_names_path=features_file,
        threshold_path=threshold_file,
    )
    ml.load()

    assert ml.model is None


# ================================ PREDICT =============================


# test predict() (model nn chargé, nb features diff, seuil par défaut ou opt)
# predict - CAS modele non chargé
def test_predict_model_not_loaded():
    ml = MLModel()

    with pytest.raises(ValueError):
        ml.predict([1.0, 2.0, 3.0])


# =======================================================================


# predict - CAS nombre de features différents
def test_predict_wrong_feature_length(mock_catboost):
    ml = MLModel()
    ml.model = mock_catboost
    ml.features_names = ["f1", "f2", "f3"]
    ml.threshold = 0.6

    with pytest.raises(ValueError):
        ml.predict([1.0, 2.0])


# =======================================================================


# predict - CAS seuil opt
def test_predict_with_threshold(mock_catboost):
    ml = MLModel()
    ml.model = mock_catboost
    ml.features_names = ["f1", "f2", "f3"]
    ml.threshold = 0.6

    pred, conf, label = ml.predict([1.0, 2.0, 3.0])

    assert pred == 1.0
    assert conf == 0.7
    assert label == "Démissionaire"


# =======================================================================


# predict - CAS seuil par défaut
def test_predict_without_threshold(mock_catboost):
    ml = MLModel()
    ml.model = mock_catboost
    ml.features_names = ["f1", "f2", "f3"]
    ml.threshold = None

    pred, conf, label = ml.predict([1.0, 2.0, 3.0])

    assert pred == 1.0
    assert conf == 0.7
    assert label == "Démissionaire"


# ==================== GET FEATURE IMPORTANCE ================================


# test get_feature_importance()
# (model non chargé, features_names non défini, top_n)
def test_get_feature_importance(mock_catboost):
    ml = MLModel()
    ml.model = mock_catboost
    ml.features_names = ["f1", "f2", "f3"]

    top_features = ml.get_feature_importance(top_n=3)

    assert len(top_features) == 3  # verifie que le nb d'element est correct
    assert top_features[0][0] in ml.features_names  # vérif nom feature juste
    assert isinstance(top_features[0][1], float)  # verif importance est float
