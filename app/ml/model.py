"""
Charge le modèle catboost optimisé pré-entrainé et gère ses commandes
"""

import logging  # sert a la gestion de logs de python
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd  # noqa: F401
from catboost import CatBoostClassifier

# Récupère/crée logger avec nom du module courant (ex: __name__="model")
logger = logging.getLogger(__name__)

# Chemin à charger:
# __file__ : chemin du fichier en cours d'exe (model.py)
# resolve(): convertie en chemin absolu + parent: donne le dossier du parent
BASE_DIR = Path(__file__).resolve().parent / "model/datas/results"


class MLModel:
    """Classe pour gérer le modèle ML"""

    def __init__(
        self,
        model_path: str = BASE_DIR / "model/best_model.cbm",
        features_names_path: str = BASE_DIR / "features_names/features_names.pkl",
        threshold_path: str | None = BASE_DIR / "threshold_opt/thresh_opt.pkl",
    ):
        self.model_path = model_path
        self.features_names_path = features_names_path
        self.threshold_path = threshold_path

        self.model: CatBoostClassifier | None = (
            None  # Instance CBC ou None==> utilité doc
        )
        self.features_names: List[str] | None = None
        self.threshold: float | None = None
        self.classes = ["Employé", "Démissionaire"]  # Codé en dur

    def get_model_info(self) -> dict:
        """
        Compile la métadatas du modèle

        SORTIES:
        model_type: str
        n_features: int
        classes: List[str]
        threshold: float | None
        """

        if self.model is None:
            raise ValueError("Modèle non chargé")

        return {
            "model_type": type(self.model).__name__,
            "n_features": len(self.features_names),
            "classes": self.classes,
            "threshold": self.threshold,
        }

    def load(self) -> None:

        # Charge le modele
        if not self.model_path.exists():
            logger.error(f"Le fichier modèle n'existe pas: {self.model_path}")
            return
        self.model = (
            CatBoostClassifier()
        )  # Instantiation de CBC ==> Création de l'objet
        self.model.load_model(self.model_path)
        logger.info(f" Modèle chargé depuis {self.model_path}")

        # Charge la liste des features
        if not self.features_names_path.exists():
            logger.error(f"Fichier features absent: {self.features_names_path}")
            return
        with open(self.features_names_path, "rb") as fn:
            self.features_names = pickle.load(fn)
            logger.info(f"features chargées via {self.features_names_path}")

        # Charge le seuil de validation
        if not self.threshold_path.exists():
            logger.error(f"Fichier seuil absent: {self.threshold_path}")
            return
        with open(self.threshold_path, "rb") as t:
            self.threshold = pickle.load(t)
            logger.info(f"Seuil chargé via {self.threshold_path}")

    def predict(self, features: List[float]) -> Tuple[float, float, str]:
        """
        Réalise une prédiction

        ENTREES:
        features: Liste des features
        SORTIES:
        Tuple (prediction, confiance, classe)
        """

        if self.model is None:
            raise ValueError("Modèle non chargé")

        if self.features_names and len(features) != len(self.features_names):
            raise ValueError(f"Nb features diff de {len(self.features_names)}")

        # Reshape liste features 1D en array 2D (1, n_features)
        features_array = np.array(features).reshape(1, -1)

        # Prédiction
        probas = self.model.predict_proba(features_array)[0]
        if self.threshold is None:
            prediction = self.model.predict(features_array)[0]
            confidence = float(np.max(probas))
        else:
            confidence = probas[1]
            prediction = int(confidence >= self.threshold)

        class_name = self.classes[int(prediction)]
        # conversion float pour compatibilité json
        prediction_value = float(prediction)

        return prediction_value, confidence, class_name

    def get_feature_importance(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Retourne les top_n features les plus influentes
        """

        if self.model is None:
            raise ValueError("Modèle non chargé")

        if not self.features_names:
            raise ValueError("Liste des features non définie")

        importances = self.model.get_feature_importance()
        importances_dict = dict(zip(self.features_names, importances))

        top_features = sorted(
            importances_dict.items(), key=lambda item: abs(item[1]), reverse=True
        )

        return top_features[:top_n]


# Instance globale et unique
ml_model = MLModel()
