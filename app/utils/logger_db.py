"""
Module de gestion de la journalisation (logging) en base de données.

Ce module fournit les outils nécessaires pour suivre le cycle de vie d'une requête HTTP.
Il permet de mesurer la performance (temps de réponse), de capturer les codes de statut
et d'établir un lien de parenté entre un log technique et une prédiction métier.
Ces données sont essentielles pour le monitoring et l'audit du service.
"""

# imports
import time

from sqlalchemy.orm import Session

from app.db.models_db import RequestLog

# ============== Initalise le log ======================


def init_log(db: Session, endpoint: str) -> RequestLog:
    """
    Initialise une entrée de log dans la table 'request_logs'.

    Crée l'objet de log au début de la requête. L'utilisation de `db.flush()`
    permet de récupérer l'ID auto-incrémenté généré par PostgreSQL sans pour autant
    clore la transaction SQL, permettant ainsi d'associer cet ID à d'autres opérations.

    Args:
        db (Session): Session SQLAlchemy active.
        endpoint (str): Le chemin de l'URL sollicité (ex: '/predict').

    Returns:
        RequestLog: L'instance du log nouvellement créée.
    """
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
    """
    Finalise et persiste le log en base de données.

    Cette fonction calcule la latence totale de la requête, met à jour le code
    de statut final (200, 422, 500, etc.) et valide la transaction (commit).

    Args:
        db (Session): Session SQLAlchemy active.
        log_obj (RequestLog): L'objet log initialisé par `init_log`.
        start_time (float): Le timestamp de début (provenant de `time.time()`).
        status_code (int, optional): Le code HTTP final. Si None, conserve la valeur initiale.
        prediction_id (str, optional): L'identifiant unique (hash) de la prédiction associée.
    """
    log_obj.status_code = status_code
    log_obj.response_time_ms = (time.time() - start_time) * 1000
    if prediction_id:
        log_obj.prediction_id = prediction_id

    db.commit()


# =============== Link log avec prediction =================


def link_log(db: Session, log_id: int, prediction_id: str):
    """
    Établit un lien a posteriori entre un log technique et un enregistrement métier.

    Cette fonction est utile car elle garantie l'intégrité de la
    relation entre les tables 'request_logs' et 'predictions'.

    Args:
        db (Session): Session SQLAlchemy active.
        log_id (int): Identifiant numérique du log.
        prediction_id (str): Hash SHA-256 de la prédiction.
    """
    log_entry = db.get(RequestLog, log_id)
    if log_entry:
        log_entry.prediction_id = prediction_id
        db.flush()
