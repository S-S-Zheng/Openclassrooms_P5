"""
Génère un id unique à partir du hashing sur les features et sauvegarde la requête
"""

# imports

from sqlalchemy.orm import Session

from app.api.models_db import PredictionRecord
from app.utils.hash_id import generate_feature_hash

# ==========================


def save_prediction(db: Session, features: dict, pred_data: tuple):
    """Gère la génération d'ID et l'enregistrement."""
    prediction, confidence, class_name = pred_data

    # Generation ID de 12 caractères hexadecimaux
    request_id = generate_feature_hash(features)

    new_record = PredictionRecord(
        id=request_id,
        inputs=features,
        prediction=int(prediction),
        confidence=float(confidence),
        class_name=class_name,
    )

    try:
        db.add(new_record)
        db.commit()
        db.refresh(new_record)
        return new_record

    except Exception as e:
        db.rollback()
        # On relance l'erreur pour que l'API puisse la gérer ou la logger
        raise e
