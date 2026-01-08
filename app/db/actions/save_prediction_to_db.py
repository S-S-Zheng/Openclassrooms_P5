"""
Génère un id unique à partir du hashing sur les features et sauvegarde la requête
"""

# imports

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models_db import PredictionRecord
from app.utils.hash_id import generate_feature_hash
from app.utils.logger_db import link_log

# ==========================


def save_prediction(db: Session, features: dict, pred_data: tuple, log_id: int):
    """Sauvegarder la nouvelle donnée"""
    prediction, confidence, class_name = pred_data

    # Generation ID de caractères hexadecimaux
    request_id = generate_feature_hash(features)

    try:
        # Si deux requetes par ex étaient lancé en meme temps, get_prediction se ferait avoir
        existing = db.scalars(
            select(PredictionRecord).where(PredictionRecord.id == request_id)
        ).first()

        if not existing:
            new_record = PredictionRecord(
                id=request_id,
                inputs=features,
                prediction=int(prediction),
                confidence=float(confidence),
                class_name=class_name,
                **features  # On unpack les features
            )
            db.add(new_record)
            db.flush()

        # Liaison avec le log pour la traça
        link_log(db, log_id, request_id)

        return request_id

    except Exception as e:
        db.rollback()
        # On relance l'erreur pour que l'API puisse la gérer ou la logger
        raise e
