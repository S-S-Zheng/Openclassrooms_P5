"""
Test des routes de l'API
"""

# Imports

# ========================= PREDICT ==========================


# Happy path
def test_predict_success(client, mock_ml_model):
    mock_ml_model(should_fail=False)

    payload = {"features": [1.0, 2.0, 3.0]}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1.0
    assert data["confidence"] == 0.85
    assert data["class_name"] == "Démissionaire"


# Echec ValueError (requete invalide == 400 )
def test_predict_value_error(client, mock_ml_model):
    mock_ml_model(should_fail=True, error_type="value")

    payload = {"features": [1.0, 2.0, 3.0]}
    response = client.post("/predict", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"] == "Modèle non chargé"


# Echec erreur critique interne (erreur serveur == 500)
def test_predict_unexpected_error(client, mock_ml_model):
    mock_ml_model(should_fail=True, error_type="exception")

    payload = {"features": [1.0, 2.0, 3.0]}
    response = client.post("/predict", json=payload)

    assert response.status_code == 500
    assert response.json()["detail"] == "Erreur interne critique"


# ========================= FEATURE IMPORTANCE ==========================


# Happy path
def test_feature_importance_success(client, mock_ml_model):
    mock_ml_model(should_fail=False, endpoint="feature_importance")

    response = client.get("/feature-importance?top_n=5")

    assert response.status_code == 200
    data = response.json()

    assert "top_features" in data
    assert len(data["top_features"]) == 5
    assert isinstance(data["top_features"][0][0], str)
    assert isinstance(data["top_features"][0][1], float)


# Tests redondant des erreurs mais obligatoire pour respecter % coverage
# ==> on test le contrat de chaque endpoint != erreur métier
# Echec ValueError (requete invalide == 400 )
def test_feature_importance_value_error(client, mock_ml_model):
    mock_ml_model(should_fail=True, error_type="value", endpoint="feature_importance")

    response = client.get("/feature-importance?top_n=5")

    assert response.status_code == 400
    assert response.json()["detail"] == "Modèle non chargé"


# Echec erreur critique interne (erreur serveur == 500)
def test_feature_importance_unexpected_error(client, mock_ml_model):
    mock_ml_model(
        should_fail=True, error_type="exception", endpoint="feature_importance"
    )

    response = client.get("/feature-importance?top_n=5")

    assert response.status_code == 500
    assert response.json()["detail"] == "Erreur interne critique"


# ========================= METADATAS ===========================


# Happy path
def test_model_info_success(client, mock_ml_model):
    mock_ml_model(should_fail=False, endpoint="model_info")

    response = client.get("/model-info")

    assert response.status_code == 200
    data = response.json()

    assert data["model_type"] == "CatBoostClassifier"
    assert data["n_features"] == 5
    assert data["classes"] == ["Employé", "Démissionaire"]
    assert data["threshold"] == 0.6


# Echec ValueError (requete invalide == 400 )
def test_model_info_value_error(client, mock_ml_model):
    mock_ml_model(should_fail=True, error_type="value", endpoint="model_info")

    response = client.get("/model-info")

    assert response.status_code == 400
    assert response.json()["detail"] == "Modèle non chargé"


# Echec erreur critique interne (erreur serveur == 500)
def test_model_info_unexpected_error(client, mock_ml_model):
    mock_ml_model(should_fail=True, error_type="exception", endpoint="model_info")

    response = client.get("/model-info")

    assert response.status_code == 500
    assert response.json()["detail"] == "Erreur interne critique"
