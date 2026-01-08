"""
Router pour le endpoint /predict, permet d'associer la requete API \n
a la méthode de la classe MLModel tel quel:\n
POST /predict ==> route.predict() ==> MLModel.predict()
"""

# ====================== Imports ========================
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.api.schemas import PredictionInput, PredictionOutput
from app.db.actions.get_prediction_from_db import get_prediction
from app.db.actions.save_prediction_to_db import save_prediction
from app.db.database import get_db
from app.utils.logger_db import closing_log, init_log

router = APIRouter(prefix="/predict", tags=["Prediction"])

# ===================== Initialisation du modele =========================


@router.post("/", response_model=PredictionOutput)
def predict(request: Request, payload: PredictionInput, db: Session = Depends(get_db)):
    """
    Endpoint de prédiction du modèle ML.
    """
    # On initialise le temps et le log
    start_time = time.time()
    log_entry = init_log(db, "/predict")

    # Récupération de l'instance du modèle depuis le state de l'application
    model_instance = getattr(request.app.state, "model", None)
    if model_instance is None:
        closing_log(db, log_entry, start_time, status_code=503)
        raise HTTPException(status_code=503, detail="Modèle non chargé sur le serveur")

    # Garde-fou pour empêcher de save plusieurs fois même requete dans db
    cached = get_prediction(db, payload.features)
    if cached:
        closing_log(db, log_entry, start_time, prediction_id=cached.id)
        return PredictionOutput(
            prediction=cached.prediction,
            confidence=cached.confidence,
            class_name=cached.class_name,
        )

    try:
        prediction, confidence, class_name = model_instance.predict(payload.features)
        # Sauvegarde de la requete + ID pour log
        request_id = save_prediction(
            db,
            payload.features,
            (prediction, confidence, class_name),
            log_id=log_entry.id,
        )
        closing_log(db, log_entry, start_time, prediction_id=request_id)
    except ValueError as exc:
        db.rollback()
        closing_log(db, log_entry, start_time, status_code=422)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as e:
        db.rollback()
        closing_log(db, log_entry, start_time, status_code=500)
        raise HTTPException(status_code=500, detail=str(e))

    return PredictionOutput(
        prediction=prediction,
        confidence=confidence,
        class_name=class_name,
    )
