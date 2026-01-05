"""
Interroge la base de donnée suivant l'ID afin s'assurer que la requete n'a jamais été faite
"""

# imports

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models_db import PredictionRecord
from app.utils.hash_id import generate_feature_hash


# ==========================
# Garde-fou pour empêcher de save plusieurs fois même requete dans db
def get_prediction(db: Session, features: dict):
    """Vérifie si une prédiction identique existe déjà."""
    # On recalcul le hash des features recues
    feature_id = generate_feature_hash(features)

    return db.scalars(
        select(PredictionRecord).where(PredictionRecord.id == feature_id)
    ).first()
