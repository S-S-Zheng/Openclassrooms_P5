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
def test_prediction_input_valid(func_sample):
    obj = PredictionInput(features=func_sample["features"])

    assert obj.features == func_sample["features"]
    assert isinstance(obj.features["age"], int)
    assert isinstance(obj.features["genre"], str)
    assert isinstance(obj.features["augementation_salaire_precedente"], float)


# features manquante
def test_prediction_input_missing_features():
    with pytest.raises(ValidationError):
        PredictionInput()


# type de feature incorrect
def test_prediction_input_wrong_type():
    with pytest.raises(ValidationError):
        PredictionInput(features={"age": [30], "genre": "m", "revenu_mensuel": 2000})


# Champs obligatoires manquant
@pytest.mark.parametrize("mandatory_features", ["age", "genre", "revenu_mensuel"])
def test_prediction_mandatory_input_missing_business_rules(
    mandatory_features, func_sample
):
    # On test que la suppr de la feature mandatory_features donne bien une erreur
    payload = func_sample["features"]
    payload.pop(mandatory_features)
    with pytest.raises(ValidationError) as excinfo:
        PredictionInput(features=payload)

    assert f"Champ obligatoire manquant : {mandatory_features}" in str(excinfo.value)


# Outliers
@pytest.mark.parametrize(
    "invalid_features, expected_error_msg",
    [
        ({"age": 150, "genre": "m", "revenu_mensuel": 2000}, "18 a 65 ans"),
        ({"age": 30, "genre": "autre", "revenu_mensuel": 2000}, "m ou f"),
        ({"age": 30, "genre": "m", "revenu_mensuel": 123}, "doit >1200"),
    ],
)
def test_prediction_input_invalid_business_rules(invalid_features, expected_error_msg):
    with pytest.raises(ValidationError) as excinfo:
        PredictionInput(features=invalid_features)

    assert expected_error_msg in str(excinfo.value)


# ===================== PredictionOutput =======================


# Happy path
def test_prediction_output_valid():
    out = PredictionOutput(
        prediction=1.0,
        confidence=0.85,
        class_name="Démissionnaire",
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
            class_name="Démissionnaire",
        )


# ===================== feature importance =====================
# On teste que top_features est bien une liste de tuples (str, float)
# capable de convertir des données qui ressemblent au type ciblé
def test_feature_importance_coercion():
    # Ici, on envoie "0.42" au lieu de 0.42
    data = {"top_features": [("age", "0.42")]}
    obj = FeatureImportanceOutput(**data)

    # Pydantic doit avoir converti la string en float
    assert obj.top_features[0][1] == 0.42
    assert isinstance(obj.top_features[0][1], float)


def test_feature_importance_invalid_structure():
    # Test d'une structure corrompue (tuple trop long)
    bad_data = {"top_features": [("age", 0.42, "extra_value")]}
    with pytest.raises(ValidationError):
        FeatureImportanceOutput(**bad_data)


# ===================== Métadatas du modèle =====================
# Test de santé pas de précision, on s'attend juste a ce que toutes les
# données soient présntes et dans avec le bon type
def test_model_info_output():
    obj = ModelInfoOutput(
        model_type="CatBoostClassifier",
        n_features=5,
        feature_names=["f1", "f2", "f3", "f4", "f5"],
        cat_features=["f1", "f2"],
        num_features=["f3", "f4", "f5"],
        classes=["Employé", "Démissionnaire"],
        threshold=0.6,
    )

    assert obj.model_type == "CatBoostClassifier"
    assert obj.n_features == 5
    assert obj.feature_names == ["f1", "f2", "f3", "f4", "f5"]
    assert obj.cat_features == ["f1", "f2"]
    assert obj.num_features == ["f3", "f4", "f5"]
    assert obj.classes == ["Employé", "Démissionnaire"]
    assert obj.threshold == 0.6


# ===================== Erreur standardisée =====================
# On garantie que l'erreur est stable et que l'API ne casse pas quand
# detail est absent
def test_error_response_optional_detail():
    err = ErrorResponse(error="Invalid input")

    assert err.detail is None
