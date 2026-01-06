"""
Router pour le endpoint /model-info, permet de connaître la métadata \n
associer au modèle utilisé: modele, nombre de features, nom des classes\n
et seuil de validation\n
GET /model-info ==> routes.model_info.
"""

# ====================== Imports ========================
from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import ModelInfoOutput

router = APIRouter(prefix="/model-info", tags=["Model informations"])

# ===================== Initialisation du modele =========================


@router.get("/", response_model=ModelInfoOutput)
def model_info(request: Request):
    """
    Endpoint pour la métadatas du modèle
    """

    # Récupération de l'instance du modèle depuis le state de l'application
    model_instance = getattr(request.app.state, "model", None)
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé sur le serveur")

    try:
        info = model_instance.get_model_info()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return info
