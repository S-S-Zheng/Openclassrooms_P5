"""
Test des routes de l'API
"""

# Imports

# ========================= PREDICT ==========================


# Happy path
def test_predict_success(client, mock_ml_model, func_sample):
    mock_ml_model(should_fail=False)

    payload = func_sample
    response = client.post("/predict/", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1.0
    assert data["confidence"] == 0.85
    assert data["class_name"] == "Démissionnaire"


def test_predict_errors(client, mock_ml_model, func_sample, error_responses):
    mock_ml_model(**error_responses["mock_args"])

    payload = func_sample
    response = client.post("/predict/", json=payload)

    assert response.status_code == error_responses["expected_status"]
    assert response.json()["detail"] == error_responses["error_msg"]


# ========================= FEATURE IMPORTANCE ==========================


# Happy path
def test_feature_importance_success(client, mock_ml_model):
    mock_ml_model(should_fail=False)

    response = client.get("/feature-importance?top_n=5")

    assert response.status_code == 200
    data = response.json()

    assert "top_features" in data
    assert len(data["top_features"]) == 5
    assert isinstance(data["top_features"][0][0], str)
    assert isinstance(data["top_features"][0][1], float)


def test_feature_importance_errors(client, mock_ml_model, error_responses):
    mock_ml_model(**error_responses["mock_args"])

    response = client.get("/feature-importance?top_n=5")

    assert response.status_code == error_responses["expected_status"]
    assert response.json()["detail"] == error_responses["error_msg"]


# ========================= METADATAS ===========================


# Happy path
def test_model_info_success(client, mock_ml_model):
    model = mock_ml_model(should_fail=False)

    response = client.get("/model-info")

    assert response.status_code == 200
    data = response.json()

    assert data["model_type"] == "CatBoostClassifier"
    assert data["n_features"] == len(model.feature_names_)
    assert data["feature_names"] == model.feature_names_
    assert data["cat_features"] == ["genre", "statut_marital"]
    assert data["num_features"] == [
        "age",
        "revenu_mensuel",
        "augementation_salaire_precedente",
    ]
    assert data["classes"] == ["Employé", "Démissionnaire"]
    assert data["threshold"] == 0.6


def test_model_info_errors(client, mock_ml_model, error_responses):
    mock_ml_model(**error_responses["mock_args"])

    response = client.get("/model-info")

    assert response.status_code == error_responses["expected_status"]
    assert response.json()["detail"] == error_responses["error_msg"]
