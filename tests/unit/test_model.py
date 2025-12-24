"""
Tests du modèle ML
"""

# import numpy as np
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
    assert ml.explainer is None


# ================================= LOAD ======================================


# test de la méthode load() (model, features,seuil et shap)
# load - CAS nominal
def test_load_model_success(tmp_path, mock_pickle, mock_catboost, mock_shap):

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
    assert ml.explainer_global is not None
    assert ml.explainer_local is not None


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


# =======================================================================


# test SHAP (explainer global et local)
# On test pas les plots car visu hors sccope tests unit ==> réduit couverture
# On veut juste s'assurrer qu'il n'y a pas d'erreurs
# explain_global - CAS nominal
def test_explain_global(mock_catboost, mock_shap, sample_features):
    ml = MLModel()
    ml.model = mock_catboost
    ml.features_names = ["f1", "f2", "f3"]
    ml.explainer_global = mock_shap

    ml.explain_global(sample_features)


# =======================================================================


# explain_global - CAS modele non chargé
def test_explain_global_model_not_loaded(sample_features):
    ml = MLModel()
    with pytest.raises(ValueError):
        ml.explain_global(sample_features)


# =======================================================================


# explain_global - CAS features_names non défini
def test_explain_global_no_features(mock_catboost, sample_features):
    ml = MLModel()
    ml.model = mock_catboost
    with pytest.raises(ValueError):
        ml.explain_global(sample_features)


# =======================================================================


# explain_local - CAS nominal
def test_explain_local(mock_catboost, mock_shap):
    ml = MLModel()
    ml.model = mock_catboost
    ml.features_names = ["f1", "f2", "f3"]
    ml.explainer_local = mock_shap
    # Dataset (n_samples=3)
    features = [1.0, 2.0, 3.0]

    ml.explain_local(features, i=0, max_display=3)
