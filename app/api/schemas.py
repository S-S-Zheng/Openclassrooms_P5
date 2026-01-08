"""
Module de définition des schémas Pydantic pour l'API.

Ce module définit les contrats d'interface (Data Transfer Objects) pour les requêtes
et les réponses de l'API FastAPI. Il inclut une couche de validation métier robuste
pour garantir que les données envoyées au modèle CatBoost respectent les plages
de valeurs attendues.

Schémas principaux :
    - PredictionInput : Données d'entrée pour l'inférence avec validation métier.
    - PredictionOutput : Résultat de la prédiction (classe et confiance).
    - FeatureImportanceOutput : Liste des variables les plus influentes.
    - ModelInfoOutput : Métadonnées et configuration du modèle ML.

| Endpoint               | Méthode ML                       |
| ---------------------- | -------------------------------- |
| `/predict`             | `MLModel.predict`                |
| `/feature-importance`  | `MLModel.get_feature_importance` |
| `/health`              | `ml.model is not None`           |
| `/model-info`          | attributs du modèle              |
"""

from typing import Dict, List, Tuple, Union

# Imports
from pydantic import BaseModel, Field, field_validator

# ========================= PREDICTION ===========================


# Schéma pour les entrées de prédiction
# L'ordre des features est garantie par feature_names dans MLModel
class PredictionInput(BaseModel):
    """
    Schéma d'entrée pour la prédiction d'attrition.

    Ce modèle valide la présence des 21 caractéristiques obligatoires requises
    par le modèle CatBoost et applique des contraintes de logique métier
    (ex: âge entre 18 et 65 ans).
    """

    features: Dict[str, Union[str, float, int]] = Field(
        ...,
        # Pydantic v2 oblige a avoir une liste
        examples=[
            {
                "age": 41,
                "genre": "f",
                "revenu_mensuel": 5993,
                "statut_marital": "célibataire",
                "poste": "cadre commercial",
                "annees_dans_le_poste_actuel": 4,
                "heure_supplementaires": "oui",
                "augementation_salaire_precedente": 11,
                "nombre_participation_pee": 0,
                "nb_formations_suivies": 0,
                "distance_domicile_travail": 1,
                "niveau_education": 2,
                "domaine_etude": "infra & cloud",
                "frequence_deplacement": "occasionnel",
                "evolution_note": 0,
                "stagnation_promo": 0.0,
                "freq_chgt_poste": 0.888889,
                "revenu_mensuel_ajuste_par_nv_hierarchique": 2996.5,
                "revenu_mensuel_par_annee_xp": 665.888889,
                "freq_chgt_responsable": 0.833333,
                "satisfaction_globale_employee": 8,
            }
        ],
    )

    @field_validator("features")
    @classmethod
    def validate_business_logic(cls, features_dict: dict):
        """
        Applique les règles de validation métier sur les caractéristiques d'entrée.

        Vérifie :
            1. La présence de tous les champs obligatoires.
            2. La cohérence des valeurs numériques (min/max).
            3. La validité des catégories pour les variables qualitatives.

        Args:
            features_dict (dict): Dictionnaire brut reçu par l'API.

        Raises:
            ValueError: Si un champ manque ou si une règle métier est violée.

        Returns:
            dict: Le dictionnaire validé.
        """
        mandatory_fields = [
            "age",
            "genre",
            "revenu_mensuel",
            "statut_marital",
            "poste",
            "annees_dans_le_poste_actuel",
            "heure_supplementaires",
            "augementation_salaire_precedente",
            "nombre_participation_pee",
            "nb_formations_suivies",
            "distance_domicile_travail",
            "niveau_education",
            "domaine_etude",
            "frequence_deplacement",
            "evolution_note",
            "stagnation_promo",
            "freq_chgt_poste",
            "revenu_mensuel_ajuste_par_nv_hierarchique",
            "revenu_mensuel_par_annee_xp",
            "freq_chgt_responsable",
            "satisfaction_globale_employee",
        ]

        for field in mandatory_fields:
            if field not in features_dict:
                raise ValueError(f"Champ obligatoire manquant : {field}")

        # Mapping
        # (min, max, msg) features numériques
        num_rules = {
            "age": (18, 65, "18 a 65 ans"),
            "revenu_mensuel": (1200, 100000, "revenu mensuel >=1200"),
            "annees_dans_le_poste_actuel": (
                0,
                65,
                "annees dans le poste >= 0 (jusqu'à 65 ans)",
            ),
            "augementation_salaire_precedente": (0, 50, "augmentation entre 0 et 50%"),
            "evolution_note": (-4, 4, "evolution de la note entre -4 et 4"),
            "satisfaction_globale_employee": (0, 16, "satisfaction entre 0 et 16"),
            "freq_chgt_responsable": (0, 1, "freq chgt responsable entre 0 et 1"),
        }
        # (choix,msg) features catégorielles
        cat_rules = {
            "heure_supplementaires": (["oui", "non"], "heure supp: oui ou non"),
            "frequence_deplacement": (
                ["aucun", "occasionnel", "frequent"],
                "freq deplacement: aucun, occasionnel ou frequent",
            ),
        }
        # Validation features numériques
        for field, (min_v, max_v, msg) in num_rules.items():
            val = features_dict.get(field)
            if val is not None:
                if not (min_v <= val <= max_v):
                    raise ValueError(msg)

        # Validation features catégorielles
        for field, (allowed, msg) in cat_rules.items():
            val = features_dict.get(field)
            if val is not None:
                if str(val).lower() not in allowed:
                    raise ValueError(msg)

        return features_dict


# Schéma pour les sorties de prédiction
class PredictionOutput(BaseModel):
    """
    Schéma de réponse après une prédiction.
    """

    prediction: float = Field(..., description="Classe prédite (0 ou 1)")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Probabilité associée à la prédiction"
    )
    class_name: str = Field(..., description="Nom lisible de la classe")


# ========================= FEATURE IMPORTANCE ===========================


# Schéma pour les importances des features
class FeatureImportanceOutput(BaseModel):
    """
    Schéma de réponse pour l'importance des variables.
    """

    top_features: List[Tuple[str, float]] = Field(
        ...,
        description="Liste décroiss des features les plus influentes (nom_feature, shapley)",
    )


# ========================= METADATAS ===========================


# Métadatas du modèle
class ModelInfoOutput(BaseModel):
    """
    Schéma de réponse détaillant la configuration interne du modèle ML.
    """

    model_type: str = Field(
        ..., description="Algorithme utilisé (ex: CatBoostClassifier)"
    )
    n_features: int = Field(..., description="Nombre total de variables d'entrée")
    feature_names: List[str] = Field(
        ..., description="Noms des variables dans l'ordre attendu"
    )
    cat_features: List[str] = Field(
        ..., description="Liste des variables catégorielles"
    )
    num_features: List[str] = Field(..., description="Liste des variables numériques")
    classes: List[str] = Field(..., description="Labels des classes de prédiction")
    threshold: float = Field(..., description="Seuil de décision optimisé")


# Erreur standardisée
class ErrorResponse(BaseModel):
    """
    Schéma standardisé pour les messages d'erreur de l'API.
    """

    error: str = Field(..., description="Type ou titre de l'erreur")
    detail: str | None = Field(
        None, description="Détails supplémentaires sur la cause de l'erreur"
    )
