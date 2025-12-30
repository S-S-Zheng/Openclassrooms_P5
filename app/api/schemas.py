"""
Modèles de schémas Pydantic pour les requêtes et réponses API

| Endpoint               | Méthode ML                       |
| ---------------------- | -------------------------------- |
| `/predict`             | `MLModel.predict`                |
| `/feature-importance`  | `MLModel.get_feature_importance` |
| `/health`              | `ml.model is not None`           |
| `/model-info`          | attributs du modèle              |
"""

from datetime import datetime  # noqa:F401
from typing import Dict, List, Tuple, Union

# Imports
from pydantic import BaseModel, Field

# ========================= PREDICTION ===========================


# Schéma pour les entrées de prédiction
# L'ordre des features est garantie par feature_names dans MLModel
class PredictionInput(BaseModel):
    features: Dict[str, Union[str, float]] = Field(
        ...,
        examples={
            "age": 0.0,
            "genre": "0",
            "revenu_mensuel": 0.0,
            "statut_marital": "0",
            "poste": "0",
            "annees_dans_le_poste_actuel": 0.0,
            "heure_supplementaires": "0",
            "augementation_salaire_precedente": 0.0,
            "nombre_participation_pee": 0.0,
            "nb_formations_suivies": 0.0,
            "distance_domicile_travail": 0.0,
            "niveau_education": 0.0,
            "domaine_etude": "0",
            "frequence_deplacement": "0",
            "evolution_note": 0.0,
            "stagnation_promo": 0.0,
            "freq_chgt_poste": 0.0,
            "revenu_mensuel_ajuste_par_nv_hierarchique": 0.0,
            "revenu_mensuel_par_annee_xp": 0.0,
            "freq_chgt_responsable": 0.0,
            "satisfaction_globale_employee": 0.0,
        },
    )


# Schéma pour les sorties de prédiction
class PredictionOutput(BaseModel):
    prediction: float = Field(..., description="Classe prédite (0 ou 1)")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Probabilité associée à la prédiction"
    )
    class_name: str = Field(..., description="Nom lisible de la classe")


# ========================= FEATURE IMPORTANCE ===========================


# Schéma pour les importances des features
class FeatureImportanceOutput(BaseModel):
    top_features: List[Tuple[str, float]] = Field(
        ...,
        description="Liste décroiss des features les plus influentes (nom_feature, shapley)",
    )


# ========================= METADATAS ===========================


# Métadatas du modèle
class ModelInfoOutput(BaseModel):
    model_type: str
    n_features: int
    feature_names: List[str]
    cat_features: List[str]
    num_features: List[str]
    classes: List[str]
    threshold: float


# Erreur standardisée
class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
