"""
Router pour le endpoint /predict, permet d'associer la requete API \n
a la méthode de la classe MLModel tel quel:
POST /predict ==> route.predict() ==> MLModel.predict()
"""

# ====================== Imports ========================
from fastapi import APIRouter, HTTPException

from app.api.schemas import PredictionInput, PredictionOutput
from app.ml.model import MLModel

router = APIRouter(prefix="/predict", tags=["prediction"])

# ===================== Initialisation du modele =========================

ml_model = MLModel()


@router.post("/", response_model=PredictionOutput)
def predict(payload: PredictionInput):
    """
    Endpoint de prédiction du modèle ML.
    """
    try:
        prediction, confidence, class_name = ml_model.predict(payload.features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictionOutput(
        prediction=prediction,
        confidence=confidence,
        class_name=class_name,
    )
