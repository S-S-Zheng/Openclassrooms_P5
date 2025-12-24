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
