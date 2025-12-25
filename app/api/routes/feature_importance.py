"""
Router pour le endpoint /feature-importance, permet d'associer la requete API \n
a la méthode de la classe MLModel tel quel:
GET /feature-importance?top_n=5 ==> routes.feature_importance(top_n=5) \n
==> MLModel.get_feature_importance(top_n=5)
"""

# ====================== Imports ========================
from fastapi import APIRouter, HTTPException, Query

from app.api.schemas import FeatureImportanceOutput
from app.ml.model import MLModel

router = APIRouter(prefix="/feature-importance", tags=["Feature Importance"])

# ===================== Initialisation du modele =========================

ml_model = MLModel()


@router.get("/", response_model=FeatureImportanceOutput)
def feature_importance(top_n: int = Query(5, ge=1)):
    try:
        top_features = ml_model.get_feature_importance(top_n=top_n)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return FeatureImportanceOutput(top_features=top_features)
