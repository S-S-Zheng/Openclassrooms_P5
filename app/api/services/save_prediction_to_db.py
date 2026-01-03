import random

from sqlalchemy.orm import Session

from app.api.models_db import PredictionRecord


def save_prediction(db: Session, features: dict, pred_data: tuple):
    """Gère la génération d'ID et l'enregistrement."""
    prediction, confidence, class_name = pred_data

    # On génère un ID unique de 10 chiffres
    request_id = str(random.randint(0, 9999999999)).zfill(10)

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
