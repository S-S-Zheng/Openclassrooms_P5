"""
Router pour le endpoint /feature-importance, permet d'associer la requete API \n
a la méthode de la classe MLModel tel quel:
GET /feature-importance?top_n=5 ==> routes.feature_importance(top_n=5) \n
==> MLModel.get_feature_importance(top_n=5)
"""

# ====================== Imports ========================
from fastapi import APIRouter, HTTPException, Query, Request

from app.api.schemas import FeatureImportanceOutput

router = APIRouter(prefix="/feature-importance", tags=["Feature Importance"])

# ===================== Initialisation du modele =========================


@router.get("/", response_model=FeatureImportanceOutput)
def feature_importance(request: Request, top_n: int = Query(5, ge=1)):
    """
    Endpoint de feature importance du modèle ML.
    """
    # Récupération de l'instance du modèle depuis le state de l'application
    model_instance = getattr(request.app.state, "model", None)
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé sur le serveur")

    try:
        top_features = model_instance.get_feature_importance(top_n=top_n)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return FeatureImportanceOutput(top_features=top_features)
