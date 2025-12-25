"""
Test des schemas Pydantic utilises dans l'API.
"""

# Imports
import pytest
from pydantic import ValidationError

from app.api.schemas import (
    ErrorResponse,
    FeatureImportanceOutput,
    ModelInfoOutput,
    PredictionInput,
    PredictionOutput,
)

# ===================== PredictionInput =======================


# Happy path
def test_prediction_input_valid():
    data = {"features": [1.0, 2.5, 3.3]}
    obj = PredictionInput(**data)

    assert obj.features == data["features"]


# features manquante
def test_prediction_input_missing_features():
    with pytest.raises(ValidationError):
        PredictionInput()


# type de feature incorrect
def test_prediction_input_wrong_type():
    with pytest.raises(ValidationError):
        PredictionInput(features=[1.0, "deux", 3.3])


# ===================== PredictionOutput =======================


# Happy path
def test_prediction_output_valid():
    out = PredictionOutput(
        prediction=1.0,
        confidence=0.85,
        class_name="Démissionaire",
    )

    assert out is not None


# confidence hors bornes
# décorateur parametrize permet d'executer même test plusieurs fois
# permettant de tester les deux bornes de confidence
@pytest.mark.parametrize("confidence", [-0.1, 1.5])
def test_prediction_output_invalid_confidence(confidence):
    with pytest.raises(ValidationError):
        PredictionOutput(
            prediction=1.0,
            confidence=confidence,
            class_name="Démissionaire",
        )


# ===================== feature importance =====================
# On teste que top_features est bien une liste de tuples (str, float)
# Pas de test d'erreur (with pytest.raises...) car couverture suffisamment stricte
def test_feature_importance_output():
    data = {"top_features": [("age", 0.42), ("salary", -0.31)]}
    obj = FeatureImportanceOutput(**data)

    assert len(obj.top_features) == 2


# ===================== Métadatas du modèle =====================
# Test de santé pas de précision, on s'attend juste a ce que toutes les
# données soient présntes et dans avec le bon type
def test_model_info_output():
    obj = ModelInfoOutput(
        model_type="CatBoostClassifier",
        n_features=12,
        classes=["Employé", "Démissionaire"],
        threshold=0.6,
    )

    assert obj.model_type == "CatBoostClassifier"
    assert obj.n_features == 12
    assert obj.classes == ["Employé", "Démissionaire"]
    assert obj.threshold == 0.6


# ===================== Erreur standardisée =====================
# On garantie que l'erreur est stable et que l'API ne casse pas quand
# detail est absent
def test_error_response_optional_detail():
    err = ErrorResponse(error="Invalid input")

    assert err.detail is None
