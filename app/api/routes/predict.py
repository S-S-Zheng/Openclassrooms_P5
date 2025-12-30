"""
Router pour le endpoint /predict, permet d'associer la requete API \n
a la méthode de la classe MLModel tel quel:
POST /predict ==> route.predict() ==> MLModel.predict()
"""

# ====================== Imports ========================
from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import PredictionInput, PredictionOutput

router = APIRouter(prefix="/predict", tags=["Prediction"])

# ===================== Initialisation du modele =========================


@router.post("/", response_model=PredictionOutput)
def predict(request: Request, payload: PredictionInput):
    """
    Endpoint de prédiction du modèle ML.
    """
    # Récupération de l'instance du modèle depuis le state de l'application
    model_instance = getattr(request.app.state, "model", None)
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé sur le serveur")

    try:
        prediction, confidence, class_name = model_instance.predict(payload.features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictionOutput(
        prediction=prediction,
        confidence=confidence,
        class_name=class_name,
    )
