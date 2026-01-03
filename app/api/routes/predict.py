"""
Router pour le endpoint /predict, permet d'associer la requete API \n
a la méthode de la classe MLModel tel quel:\n
POST /predict ==> route.predict() ==> MLModel.predict()
"""

# ====================== Imports ========================
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.api.schemas import PredictionInput, PredictionOutput
from app.api.services.get_prediction_from_db import get_prediction
from app.api.services.save_prediction_to_db import save_prediction
from app.database import get_db

router = APIRouter(prefix="/predict", tags=["Prediction"])

# ===================== Initialisation du modele =========================


@router.post("/", response_model=PredictionOutput)
def predict(request: Request, payload: PredictionInput, db: Session = Depends(get_db)):
    """
    Endpoint de prédiction du modèle ML.
    """
    # Récupération de l'instance du modèle depuis le state de l'application
    model_instance = getattr(request.app.state, "model", None)
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé sur le serveur")

    # Garde-fou pour empêcher de save plusieurs fois même requete dans db
    cached = get_prediction(db, payload.features)
    if cached:
        return PredictionOutput(
            prediction=cached.prediction,
            confidence=cached.confidence,
            class_name=cached.class_name,
        )

    try:
        prediction, confidence, class_name = model_instance.predict(payload.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Sauvegarde de la requete
    save_prediction(db, payload.features, (prediction, confidence, class_name))

    return PredictionOutput(
        prediction=prediction,
        confidence=confidence,
        class_name=class_name,
    )
