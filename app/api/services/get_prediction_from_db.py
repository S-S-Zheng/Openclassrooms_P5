from sqlalchemy.orm import Session

from app.api.models_db import PredictionRecord


# Garde-fou pour empêcher de save plusieurs fois même requete dans db
def get_prediction(db: Session, features: dict):
    """Vérifie si une prédiction identique existe déjà."""
    return (
        db.query(PredictionRecord).filter(PredictionRecord.inputs == features).first()
    )
