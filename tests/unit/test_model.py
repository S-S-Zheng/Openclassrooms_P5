"""
Suite de tests unitaires pour la classe MLModel.

Ce module valide le comportement interne du wrapper de modèle CatBoost.
Il utilise intensivement le mockage pour simuler les artefacts du modèle
(fichiers .cbm, .pkl) et l'objet CatBoostClassifier lui-même, garantissant
que la logique de chargement, de prédiction et d'analyse est robuste
face à diverses configurations et erreurs de fichiers.
"""

import logging

import numpy as np  # noqa: F401
import pandas as pd
import pytest

from app.ml.model import MLModel

# =============================== INIT =======================================


# test methode init() ()
def test_init(ml_model):
    """Vérifie l'état initial par défaut d'une nouvelle instance MLModel."""
    assert ml_model.model is None
    assert ml_model.feature_names is None
    assert ml_model.cat_features is None
    assert ml_model.num_features is None
    assert ml_model.threshold is None
    assert ml_model.classes == ["Employé", "Démissionnaire"]


# ======================== GET MODEL INFO =================================


# Happy path + threshold limite
@pytest.mark.parametrize("test_threshold", [0.6, None])
def test_get_model_info(mock_catboost, test_threshold, ml_model):
    """
    Vérifie la compilation correcte des métadonnées du modèle.
    Teste la structure du dictionnaire retourné avec et sans seuil personnalisé.
    """
    ml_model.model = mock_catboost
    ml_model.feature_names = ml_model.model.feature_names_
    cat_idx = ml_model.model.get_cat_feature_indices()
    ml_model.cat_features = [ml_model.feature_names[i] for i in cat_idx]
    ml_model.num_features = [
        name for name in ml_model.feature_names if name not in ml_model.cat_features
    ]
    ml_model.classes = ["Employé", "Démissionnaire"]
    ml_model.threshold = test_threshold

    model_info = ml_model.get_model_info()

    assert isinstance(model_info, dict)
    # Rq: create_autospec(catboost.CatBoostClassifier) implique que
    # le nom du type doit être celui de la classe mockée
    assert model_info["model_type"] == "CatBoostClassifier"
    assert model_info["n_features"] == 5
    assert model_info["feature_names"] == ["f1", "f2", "f3", "f4", "f5"]
    assert model_info["cat_features"] == ["f1", "f2"]
    assert model_info["num_features"] == ["f3", "f4", "f5"]
    assert model_info["classes"] == ["Employé", "Démissionnaire"]
    # Avec le décorateur on test les deux valeurs
    assert model_info["threshold"] == test_threshold


# Echec de chargement du modele
def test_get_model_info_not_loaded(ml_model):
    """Vérifie qu'une erreur est levée si l'on demande les infos d'un modèle non chargé."""
    ml_model.model = None

    with pytest.raises(ValueError):
        ml_model.get_model_info()


# ================================= LOAD ======================================


# test de la méthode load() (model, features, seuil)
# load - CAS nominal
def test_load_model_success(
    tmp_path, mock_pickle, mock_catboost, monkeypatch, ml_model
):
    """
    Teste le scénario nominal de chargement complet du modèle.
    Valide l'intégration entre le binaire CatBoost, les noms de features et le seuil.
    """
    model_file = tmp_path / "model.cbm"
    features_file = tmp_path / "feature_names.pkl"
    threshold_file = tmp_path / "thresh_opt.pkl"

    model_file.touch()
    features_file.touch()
    threshold_file.touch()

    # On utilise monkeypatch pour que MLModel utilise notre mock au lieu de la vraie classe
    monkeypatch.setattr("app.ml.model.CatBoostClassifier", lambda: mock_catboost)

    # test les chemins ici via des fichiers temporaires
    ml = MLModel(
        model_path=model_file,
        feature_names_path=features_file,
        threshold_path=threshold_file,
    )

    ml.load()

    assert ml.model is not None
    assert isinstance(ml.feature_names, list)
    assert ml.feature_names == mock_catboost.feature_names_
    assert ml.cat_features == ["f1", "f2"]
    assert ml.num_features == ["f3", "f4", "f5"]
    assert isinstance(ml.threshold, float)
    assert ml.threshold == 0.6


# =======================================================================


# test du fallback pickle
def test_load_model_fallback_pickle(
    tmp_path, mock_pickle, mock_catboost, monkeypatch, caplog
):
    """
    Vérifie le mécanisme de repli (fallback) via Pickle.
    Si le modèle CatBoost ne contient pas les noms de features, ils doivent être
    chargés depuis le fichier de sauvegarde externe.
    """
    # On simule un modèle qui n'a pas enregistré les noms
    mock_catboost.feature_names_ = None
    monkeypatch.setattr("app.ml.model.CatBoostClassifier", lambda: mock_catboost)

    model_file = tmp_path / "model.cbm"
    features_file = tmp_path / "feature_names.pkl"
    threshold_file = tmp_path / "thresh_opt.pkl"

    model_file.touch()
    features_file.touch()
    threshold_file.touch()

    ml = MLModel(
        model_path=model_file,
        feature_names_path=features_file,
        threshold_path=threshold_file,
    )

    with caplog.at_level(logging.WARNING):
        ml.load()

    # Ici, ml.feature_names doit provenir de mock_pickle car le modèle était vide
    assert ml.feature_names == ["f1", "f2", "f3", "f4", "f5"]
    assert ml.cat_features == ["f1", "f2"]
    assert "Le modèle n'a pas sauvegardés le nom des features" in caplog.text


# =======================================================================


# Verif que si c'est feature_names est un dict (pkl) on le convertisse bien en liste
def test_load_model_fallback_pickle_load_dict(
    tmp_path, mock_pickle, mock_catboost, monkeypatch
):
    """Vérifie que le chargement des noms de features gère le format dictionnaire."""
    mock_catboost.feature_names_ = None
    monkeypatch.setattr("app.ml.model.CatBoostClassifier", lambda: mock_catboost)

    dict_test = {"feature_names": ["f11", "f12", "f13", "f14", "f15"]}
    # on construit artificiellement un dictionnaire feature_names
    monkeypatch.setattr("pickle.load", lambda file: dict_test)

    model_file = tmp_path / "model.cbm"
    features_file = tmp_path / "feature_names.pkl"
    threshold_file = tmp_path / "thresh_opt.pkl"

    model_file.touch()
    features_file.touch()
    threshold_file.touch()

    ml = MLModel(
        model_path=model_file,
        feature_names_path=features_file,
        threshold_path=threshold_file,
    )

    ml.load()

    assert isinstance(ml.feature_names, list)
    assert ml.feature_names == ["f11", "f12", "f13", "f14", "f15"]


# =======================================================================


# load - CAS model manquant
def test_load_model_failed(
    tmp_path,
):
    """Vérifie que l'instance reste à None si le fichier binaire du modèle est introuvable."""
    model_file = tmp_path / "model.cbm"
    features_file = tmp_path / "feature_names.pkl"
    threshold_file = tmp_path / "thresh_opt.pkl"

    ml = MLModel(
        model_path=model_file,
        feature_names_path=features_file,
        threshold_path=threshold_file,
    )
    ml.load()

    assert ml.model is None


# =======================================================================


# fichier liste des features absente
def test_load_feature_names_absent(tmp_path, caplog, mock_catboost, monkeypatch):
    """Vérifie le comportement et la journalisation d'erreur si le fichier de features manque."""
    # On simule un modèle qui n'a pas enregistré les noms
    mock_catboost.feature_names_ = None
    monkeypatch.setattr("app.ml.model.CatBoostClassifier", lambda: mock_catboost)

    features_file = tmp_path / "feature_names.pkl"

    ml = MLModel(feature_names_path=features_file)
    with caplog.at_level(logging.ERROR):
        ml.load()

    assert f"Fichier features absent: {features_file}" in caplog.text


# =======================================================================


# fichier seuil de validation absent
def test_load_threshold_absent(tmp_path, caplog):
    """Vérifie le comportement et la journalisation d'erreur si le fichier de seuil manque."""
    threshold_file = tmp_path / "thresh_opt.pkl"

    ml = MLModel(threshold_path=threshold_file)

    with caplog.at_level(logging.ERROR):
        ml.load()

    assert f"Fichier seuil absent: {threshold_file}" in caplog.text


# ================================ PREDICT =============================


# test predict() (model nn chargé, nb features diff, seuil par défaut ou opt)
# predict - CAS modele non chargé
def test_predict_model_not_loaded(fake_dict, ml_model):
    """S'assure qu'une prédiction est impossible sans chargement préalable."""
    with pytest.raises(ValueError):
        ml_model.predict(fake_dict)


# =======================================================================


# predict - CAS features manquantes
def test_predict_missing_features(mock_catboost, fake_dict, ml_model):
    """Vérifie le rejet d'un dictionnaire d'entrée incomplet."""
    ml_model.model = mock_catboost
    ml_model.feature_names = ["f1", "f2", "f3", "f4", "f6"]
    ml_model.threshold = 0.6

    with pytest.raises(ValueError, match="features manquantes"):
        ml_model.predict(fake_dict)


# =======================================================================


# predict - CAS nombre de features différents
def test_predict_unexpected_features(mock_catboost, fake_dict, ml_model):
    """Vérifie le rejet si le nombre de caractéristiques fournies est incorrect."""
    ml_model.model = mock_catboost
    ml_model.feature_names = ["f1", "f2", "f3", "f4"]
    ml_model.threshold = 0.6

    with pytest.raises(ValueError, match="Plus de features qu'attendues"):
        ml_model.predict(fake_dict)


# =======================================================================


# predict - CAS seuil de validation
@pytest.mark.parametrize("thresh", [0.6, None])
def test_predict_with_threshold(mock_catboost, thresh, ml_model):
    """
    Valide la logique de prédiction binaire.
    Compare le comportement entre le seuil par défaut du modèle et le seuil optimisé.
    """
    ml_model.model = mock_catboost
    ml_model.feature_names = ml_model.model.feature_names_
    ml_model.threshold = thresh
    cat_idx = ml_model.model.get_cat_feature_indices()
    ml_model.cat_features = [ml_model.feature_names[i] for i in cat_idx]
    ml_model.num_features = [
        name for name in ml_model.feature_names if name not in ml_model.cat_features
    ]

    pred, conf, class_name = ml_model.predict(
        {"f1": "1.0", "f2": "2.0", "f3": 3.0, "f4": 4.0, "f5": 5.0}
    )
    args, _ = ml_model.model.predict_proba.call_args

    assert pred == 1.0
    assert conf == 0.7
    assert class_name == "Démissionnaire"
    assert isinstance(args[0], pd.DataFrame)
    assert list(args[0].columns) == ml_model.feature_names


# ==================== GET FEATURE IMPORTANCE ================================


# Happy path
# (model non chargé, feature_names non défini, top_n)
def test_get_feature_importance(mock_catboost, ml_model):
    """
    Vérifie l'extraction et le tri des variables les plus importantes.
    S'assure notamment de la conversion des scores en types float natifs (compatibilité JSON).
    """
    ml_model.model = mock_catboost
    ml_model.feature_names = ml_model.model.feature_names_

    top_features = ml_model.get_feature_importance(top_n=3)

    assert len(top_features) == 3  # verifie que le nb d'element est correct
    assert top_features[0][0] in ml_model.feature_names  # vérif nom feature juste
    assert isinstance(top_features[0][1], float)  # verif importance est float: shapley
    assert not isinstance(
        top_features[0][1], np.floating
    )  # JSON/SQL pas stable avec numpy.float64
    assert top_features[0][1] >= top_features[1][1]  # Vérif Décroissance des features


# =======================================================================


# Echec de chargement du modele
def test_get_feature_importance_not_loaded(ml_model):
    """Vérifie la levée d'erreur si l'analyse d'importance est appelée sur un modèle vide."""
    ml_model.model = None

    with pytest.raises(ValueError):
        ml_model.get_feature_importance()


# =======================================================================


# Echec de chargement des features
@pytest.mark.parametrize("undefined", [[], None])
def test_feature_names_not_loaded(mock_catboost, undefined, ml_model):
    """Vérifie que l'importance ne peut être calculée si les noms des variables sont absents."""
    ml_model.model = mock_catboost
    ml_model.feature_names = undefined

    with pytest.raises(ValueError):
        ml_model.get_feature_importance()
