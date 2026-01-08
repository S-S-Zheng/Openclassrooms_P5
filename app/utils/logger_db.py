"""
Centralise la création et la mise à jour des logs.
"""

# imports
import time

from sqlalchemy.orm import Session

from app.db.models_db import RequestLog

# ============== Initalise le log ======================


def init_log(db: Session, endpoint: str) -> RequestLog:
    """Initialise un log en base"""
    new_log = RequestLog(endpoint=endpoint, status_code=200)
    db.add(new_log)
    db.flush()  # Pour obtenir l'ID sans committer
    return new_log


# ================ Finalise le log =======================


def closing_log(
    db: Session,
    log_obj: RequestLog,
    start_time: float,
    status_code: int = None,
    prediction_id: str = None,
):
    """Calcule le temps, met à jour le statut et le lien de dépendance, puis commit."""
    log_obj.status_code = status_code
    log_obj.response_time_ms = (time.time() - start_time) * 1000
    if prediction_id:
        log_obj.prediction_id = prediction_id

    db.commit()


# =============== Link log avec prediction =================


def link_log(db: Session, log_id: int, prediction_id: str):
    """Fait le lien entre le log technique et l'enregistrement métier."""
    log_entry = db.get(RequestLog, log_id)
    if log_entry:
        log_entry.prediction_id = prediction_id
        db.flush()
