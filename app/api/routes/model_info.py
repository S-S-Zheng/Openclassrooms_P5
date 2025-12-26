"""
Router pour le endpoint /model-info, permet de connaître la métadata \n
associer au modèle utilisé: modele, nombre de features, nom des classes\n
et seuil de validation\n
GET /model-info ==> routes.model_info.
"""

# ====================== Imports ========================
from fastapi import APIRouter, HTTPException

from app.api.schemas import ModelInfoOutput
from app.ml.model import ml_model

router = APIRouter(prefix="/model-info", tags=["Model informations"])

# ===================== Initialisation du modele =========================


@router.get("/", response_model=ModelInfoOutput)
def model_info():
    try:
        info = ml_model.get_model_info()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return info
