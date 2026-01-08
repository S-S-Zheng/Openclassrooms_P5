"""
Module de définition du router pour l'analyse de l'importance des variables.

Ce module expose un endpoint permettant de comprendre la logique interne du modèle
en identifiant les caractéristiques (features) qui ont le plus d'influence sur
les prédictions d'attrition. Il fait le pont entre les requêtes HTTP et la
méthode de calcul d'importance du modèle CatBoost.
"""

# ====================== Imports ========================
from fastapi import APIRouter, HTTPException, Query, Request

from app.api.schemas import FeatureImportanceOutput

router = APIRouter(prefix="/feature-importance", tags=["Feature Importance"])

# ===================== Initialisation du modele =========================


@router.get("/", response_model=FeatureImportanceOutput)
def feature_importance(request: Request, top_n: int = Query(5, ge=1)):
    """
    Récupère les variables les plus influentes du modèle de prédiction.

    Cet endpoint interroge l'instance du modèle chargée en mémoire pour extraire
    les 'n' caractéristiques ayant le poids le plus élevé dans le processus de décision.
    Cela permet une explicabilité globale du modèle (Global Feature Importance).

    Args:
        request (Request): Objet requête FastAPI utilisé pour accéder au 'state' de l'application.
        top_n (int): Nombre de variables à retourner, triées par importance décroissante.
            Doit être supérieur ou égal à 1. Par défaut à 5.

    Returns:
        FeatureImportanceOutput: Un objet contenant la liste des tuples (nom_variable, score).

    Raises:
        HTTPException (503): Si l'instance du modèle n'est pas trouvée dans l'état de l'application.
        HTTPException (422): Si les paramètres de requête sont invalides pour le modèle.
        HTTPException (500): En cas d'erreur interne lors du calcul des importances.

    Note:
        Le modèle utilise généralement les valeurs SHAP ou les scores de gain d'information
        natifs de CatBoost pour classer les variables.
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
