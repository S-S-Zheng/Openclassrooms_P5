"""
Tests sur les features qui interagissent avec la DB
"""

# imports

import pytest
from sqlalchemy import create_engine, delete, func, inspect, select
from sqlalchemy.orm import Session, sessionmaker  # noqa: F401

from app.db import database
from app.db.actions.save_prediction_to_db import save_prediction
from app.db.create_db import init_db
from app.db.import_dataset_to_db import import_csv
from app.db.models_db import PredictionRecord, RequestLog
from app.utils.hash_id import generate_feature_hash
from app.utils.logger_db import init_log

# ========================== DATABASE =======================


def test_get_db(monkeypatch, TestingSession):
    """Vérifie que get_db fournit une session valide et la ferme après."""

    # On force database.SessionLocal à utiliser la session de test
    monkeypatch.setattr(database, "SessionLocal", TestingSession)

    # On teste le générateur (@contextmanager ==> on l'utilise avec 'with')
    with database.get_db_contextmanager() as session:
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


# Gestion de l'idempotence
def test_save_prediction_unique_id(db_session_for_tests, func_sample):
    """
    Vérifie que save_prediction ne crée pas de doublon si appelée deux fois
    avec les mêmes features, mais retourne le même ID tout en créant les logs.
    """
    features = func_sample["features"]
    pred_data = (1, 0.85, "Démissionnaire")

    # On instancie deux log
    log_1 = init_log(db_session_for_tests, "/predict")
    log_2 = init_log(db_session_for_tests, "/predict")

    # On injecte une premère fois la donnée qui doit bien se passer
    request_id_1 = save_prediction(
        db_session_for_tests, features, pred_data, log_id=log_1.id
    )
    # Deuxième enregistrement (mêmes données, log différent)
    request_id_2 = save_prediction(
        db_session_for_tests, features, pred_data, log_id=log_2.id
    )

    assert (
        request_id_1 == request_id_2
    )  # L'ID doit être le même (basée sur les features)
    # On vérifie qu'il n'y a pas de doublon
    count = db_session_for_tests.scalar(
        select(func.count()).select_from(PredictionRecord)
    )
    assert count == 1
    # On vérifie que les deux ID log pointent vers le même ID (features)
    assert log_1.prediction_id == request_id_1
    assert log_2.prediction_id == request_id_2


# ================================================================


# Crash en cours d'opération
def test_save_prediction_crash(db_session_broken_for_tests, func_sample):
    """
    Vérifie qu'en cas de crash lors d'une opération de sauvegarde, on rollback
    """
    features = func_sample["features"]
    pred_data = (1, 0.85, "Démissionnaire")
    log_1 = 1

    with pytest.raises(Exception) as excinfo:
        save_prediction(db_session_broken_for_tests, features, pred_data, log_id=log_1)
    assert "Unexpected Crash" in str(excinfo.value)
    # On vérifie que le rollback a bien fonctionné
    assert db_session_broken_for_tests.rollback.called


# ====================== INITIALISATION DB ===========================


# Happy path
def test_init_db(TestingEngine):
    """
    On vérifie que l'appel à init_db ne provoque pas d'erreur
    et que les tables sont bien présentes.
    """
    init_db(reset_tables=False, engine=TestingEngine)

    # Vérification physique de la présence de la table via l'inspecteur
    inspector = inspect(TestingEngine)
    tables = inspector.get_table_names()
    assert "request_logs" in tables
    assert "predictions" in tables


# =============================================================


# Reset fonctionnel
def test_init_db_reset_table(TestingEngine, func_sample, TestingSession):
    """
    On vérifie que les tables sont bien présentes sont bien nettoyées.
    """
    # On créée une session parallèle pour constituer une base non vide et
    # s'assurer que la donnée est présente
    with TestingSession() as temp_session:
        log_entry = init_log(temp_session, "/predict")
        save_prediction(
            temp_session,
            func_sample["features"],
            (0, 0.9, "Employé"),
            log_id=log_entry.id,
        )
        temp_session.commit()

    # Reset
    init_db(reset_tables=True, engine=TestingEngine)

    # Vérification physique de la présence de la table via l'inspecteur
    inspector = inspect(TestingEngine)
    tables = inspector.get_table_names()
    assert "request_logs" in tables
    assert "predictions" in tables

    # Vérifications logiques (vidage des données)
    with TestingSession() as new_session:
        count_pred = new_session.scalar(
            select(func.count()).select_from(PredictionRecord)
        )
        count_logs = new_session.scalar(select(func.count()).select_from(RequestLog))

        assert count_pred == 0
        assert count_logs == 0


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
def test_import_csv_success(monkeypatch, TestingSession, TestingEngine, fake_csv):
    """
    Vérifie que le CSV est bien lu et inséré en base avec les bons IDs et le log
    """
    # On import chemin et données
    fake_file_path, fake_data = fake_csv

    # On court-circuite get_db
    monkeypatch.setattr(database, "SessionLocal", TestingSession)

    # On importe
    import_csv(str(fake_file_path))

    with TestingSession() as session:
        # vérification métier
        records = session.scalars(select(PredictionRecord)).all()

        assert len(records) == len(fake_data["features"]["a_quitte_l_entreprise"])

        # Vérification du mapping
        classification = session.scalars(
            select(PredictionRecord).filter_by(class_name="Démissionnaire")
        ).first()

        assert classification is not None
        assert classification.prediction == 1
        assert classification.confidence == 1.0

        # Vérification du log
        log = session.scalars(
            select(RequestLog).where(RequestLog.endpoint.contains("/import"))
        ).first()

        assert log is not None
        assert log.status_code == 201
        assert log.response_time_ms > 0


# ==============================================================


# Unicité
def test_import_csv_unique(fake_csv, TestingSession, monkeypatch, TestingEngine):
    """
    Vérifie que la data hist n'est pas ré enregistrer si déjà présente
    """
    # On import chemin et données
    fake_file_path, fake_data = fake_csv

    # On court-circuite get_db
    monkeypatch.setattr(database, "SessionLocal", TestingSession)

    # On clean les tables avant de lancer le test (à cause de la fixture init)
    with TestingSession() as session:
        session.execute(delete(RequestLog))
        session.execute(delete(PredictionRecord))
        session.commit()

    # On importe deux fois, le premier devrait etre bon seulement
    import_csv(str(fake_file_path))
    import_csv(str(fake_file_path))

    with TestingSession() as session:
        # Un seul import features
        records = session.scalars(select(PredictionRecord)).all()

        assert len(records) == len(fake_data["features"]["a_quitte_l_entreprise"])

        # Mais deux logs d'import
        log = session.scalars(select(RequestLog)).all()

        assert len(log) == 2


# =============================================================


# Crash immédiat
def test_import_csv_crash_instant(db_session_broken_for_tests, fake_csv, monkeypatch):
    """
    Vérifie qu'en cas de crash avant même d'ouvrir le log, on rollback
    """
    fake_file_path, _ = fake_csv
    monkeypatch.setattr(database, "SessionLocal", lambda: db_session_broken_for_tests)

    with pytest.raises(Exception) as excinfo:
        import_csv(str(fake_file_path))

    assert "Unexpected Crash" in str(excinfo.value)
    # On vérifie que le rollback a bien fonctionné
    assert db_session_broken_for_tests.rollback.called
