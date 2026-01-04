"""
Tests sur les features qui interagissent avec la DB
"""

# imports

import pandas as pd
import pytest
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session, sessionmaker  # noqa: F401

from app import database
from app.api.models_db import PredictionRecord
from app.api.services.save_prediction_to_db import save_prediction
from app.create_db import init_db
from app.import_dataset_to_db import import_csv
from app.utils.hash_id import generate_feature_hash

# ========================== DATABASE =======================


def test_get_db(monkeypatch, TestingSession):
    """Vérifie que get_db fournit une session valide et la ferme après."""

    # On force database.SessionLocal à utiliser la session de test
    monkeypatch.setattr(database, "SessionLocal", TestingSession)

    # On teste le générateur (@contextmanager ==> on l'utilise avec 'with')
    with database.get_db() as session:
        assert isinstance(session, Session)
        assert session.is_active

    # On vérifie que la session est fermée
    assert session.get_transaction() is None


# =========================== PERSISTENCE ==========================


def test_get_prediction(client, mock_ml_model, func_sample, db_session_for_tests):
    """Vérifie que l'API utilise la DB si la donnée existe déjà."""
    # On insère manuellement une prédiction lié a fun_sample dans la db
    existing_pred = PredictionRecord(
        id=generate_feature_hash(func_sample["features"]),
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


# Happy path
def test_save_prediction(client, mock_ml_model, func_sample, db_session_for_tests):
    """Vérifie qu'une nouvelle prédiction est bien insérée en base."""
    mock_ml_model(should_fail=False)
    payload = func_sample
    expected_hash_id = generate_feature_hash(func_sample["features"])

    # Premier appel: l'API va utiliser sa propre session via Depends(get_db)
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200

    saved_pred = db_session_for_tests.scalars(
        select(PredictionRecord).filter_by(id=expected_hash_id)
    ).first()

    assert saved_pred is not None
    assert saved_pred.id == expected_hash_id
    assert saved_pred.inputs == payload["features"]
    # La prédiction doit correspondre aux résultats du mock_ml_model.predict
    assert saved_pred.class_name == "Démissionnaire"
    assert saved_pred.prediction == 1
    assert saved_pred.confidence == 0.85


# =====================================================


# Erreur interne critique (code 500)
def test_save_prediction_integrity_error(db_session_for_tests, func_sample):
    """Vérifie que save_prediction lève une exception en cas de doublon (Clé Primaire)."""
    features = func_sample["features"]
    pred_data = (1, 0.85, "Démissionnaire")
    # On injecte une premère fois la donnée qui doit bien se passer
    save_prediction(db_session_for_tests, features, pred_data)

    # On va tenter d'injecter une seconde fois la meme donnée (même ID) et provoqué code 500
    with pytest.raises(Exception):
        save_prediction(db_session_for_tests, features, pred_data)


# ====================== INITIALISATION DB ===========================


# Happy path
def test_init_db(db_session_for_tests, TestingEngine):
    """
    On vérifie que l'appel à init_db ne provoque pas d'erreur
    et que les tables sont bien présentes.
    """
    init_db(reset_tables=False, engine=TestingEngine)

    # Vérification physique de la présence de la table via l'inspecteur
    inspector = inspect(TestingEngine)
    assert "predictions" in inspector.get_table_names()


# =============================================================


# Reset fonctionnel
def test_init_db_reset_table(TestingEngine, func_sample, TestingSession):
    """
    On vérifie que les tables sont bien présentes sont bien nettoyées.
    """
    # On créée une session parallèle pour constituer une base non vide et
    # s'assurer que la donnée est présente
    with TestingSession() as temp_session:
        save_prediction(temp_session, func_sample["features"], (0, 0.9, "Employé"))

    init_db(reset_tables=True, engine=TestingEngine)

    # Vérification physique de la présence de la table via l'inspecteur
    inspector = inspect(TestingEngine)
    assert "predictions" in inspector.get_table_names()


# =================================================================


# Echec de création de base
def test_init_db_failed_init():
    """
    On vérifie que l'echec de connexion à la base renvoie une exception
    """
    fake_engine = create_engine("postgresql://user:wrong@localhost:9999/fake")

    with pytest.raises(Exception):
        init_db(reset_tables=False, engine=fake_engine)


# ======================= IMPORTATION DE DONNEES =======================


# Happy path
def test_import_csv_success(monkeypatch, tmp_path, TestingSession, TestingEngine):
    """
    Vérifie que le CSV est bien lu et inséré en base avec les bons IDs
    """
    # faux chemin avec fichier csv
    fake_file = tmp_path / "hist_datas"
    fake_file.mkdir()
    fake_file_path = fake_file / "test_data.csv"
    fake_data = {
        "features": {
            "age": [20, 50],
            "genre": ["f", "m"],
            "revenue_mensuel": [1500, 8000],
            "a_quitte_l_entreprise": [1, 0],
        }
    }
    pd.DataFrame(fake_data["features"]).to_csv(fake_file_path, index=False)

    # On court-circuite get_db
    monkeypatch.setattr(database, "SessionLocal", TestingSession)

    # On importe
    import_csv(str(fake_file_path))

    # On ouvre une autre session avec TestingEngine pour vérifier le nombre d'observation
    # on ouvre une nouvelle session pour éviter un eventuel "session is closed"
    Session = sessionmaker(bind=TestingEngine)
    with Session() as session:
        records = session.scalars(select(PredictionRecord)).all()

        assert len(records) == len(fake_data["features"]["a_quitte_l_entreprise"])

        # Vérification du mapping
        classification = session.scalars(
            select(PredictionRecord).filter_by(class_name="Démissionnaire")
        ).first()

        assert classification is not None
        assert classification.prediction == 1
        assert classification.confidence == 1.0
