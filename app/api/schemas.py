"""
Modèles de schémas Pydantic pour les requêtes et réponses API

| Endpoint               | Méthode ML                       |
| ---------------------- | -------------------------------- |
| `/predict`             | `MLModel.predict`                |
| `/feature-importance`  | `MLModel.get_feature_importance` |
| `/health`              | `ml.model is not None`           |
| `/model-info`          | attributs du modèle              |
"""

from typing import List, Tuple

# Imports
from pydantic import BaseModel, Field

# ========================= PREDICTION ===========================


# Schéma pour les entrées de prédiction
# L'ordre des features est garantie par features_names dans MLModel
class PredictionInput(BaseModel):
    features: List[float]


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
    top_features: List[Tuple[str, float]]


# ========================= METADATAS ===========================


# Métadatas du modèle
class ModelInfoOutput(BaseModel):
    model_type: str
    n_features: int
    classes: List[str]
    threshold: float | None


# Erreur standardisée
class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
