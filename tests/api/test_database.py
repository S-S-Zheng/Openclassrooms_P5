from app.api.models_db import PredictionRecord

# =========================== PERSISTENCE ==========================


def test_get_prediction(client, mock_ml_model, func_sample, db_session_for_tests):
    """Vérifie que l'API utilise la DB si la donnée existe déjà."""
    # On insère manuellement une prédiction lié a fun_sample dans la db
    existing_pred = PredictionRecord(
        id="9999999999",
        inputs=func_sample["features"],
        prediction=0,
        confidence=0.99,
        class_name="Employé",
        model_version="v1_test",
    )
    db_session_for_tests.add(existing_pred)
    db_session_for_tests.commit()

    # On configure le ML pour qu'il réponde l'inverse de la DB
    mock_ml_model(should_fail=False)
    response = client.post("/predict/", json=func_sample)

    data = response.json()
    # On confirme que la DB a été utilisée
    assert data["prediction"] == existing_pred.prediction
    assert data["class_name"] == existing_pred.class_name


# ======================== REMPLISSAGE DES TABLES ===========================


def test_save_prediction(client, mock_ml_model, func_sample, db_session_for_tests):
    """Vérifie qu'une nouvelle prédiction est bien insérée en base."""
    # 1. Configuration
    mock_ml_model(should_fail=False)
    payload = func_sample

    # 2. Action : Premier appel
    # L'API va utiliser sa propre session via Depends(get_db)
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200

    # 3. Vérification : La ligne existe en DB
    # On vérifie avec NOTRE session de test si la donnée est là
    # Comme l'API et le test partagent le même engine/transaction, le test voit l'insert
    saved_pred = db_session_for_tests.query(PredictionRecord).first()

    assert saved_pred is not None
    assert saved_pred.class_name == "Démissionaire"
    assert saved_pred.inputs == payload["features"]
    assert saved_pred.prediction == 1
    assert saved_pred.confidence == 0.85


# ======================
