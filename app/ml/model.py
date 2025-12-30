"""
Charge le modèle catboost optimisé pré-entrainé et gère ses commandes
"""

import logging  # sert a la gestion de logs de python
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
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
        feature_names_path: str = BASE_DIR / "feature_names/feature_names.pkl",
        threshold_path: str | None = BASE_DIR / "threshold_opt/thresh_opt.pkl",
    ):
        self.model_path = model_path
        self.feature_names_path = feature_names_path
        self.threshold_path = threshold_path

        self.model: CatBoostClassifier | None = (
            None  # Instance CBC ou None==> utilité doc
        )
        self.feature_names: List[str] | None = None
        self.cat_features: List[str] | None = None
        self.num_features: List[str] | None = None
        self.threshold: float | None = None
        self.classes = ["Employé", "Démissionaire"]  # Codé en dur

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
        if self.model.feature_names_:
            self.feature_names = self.model.feature_names_
            logger.info("features chargées depuis le modèle")
        else:
            # Cas où Pool n'a pas été utilisé, CBC garde en mémoire alors juste
            # indices catégorielles sans feature_names_ (Fallback)
            logger.warning("Le modèle n'a pas sauvegardés le nom des features")
            if not self.feature_names_path.exists():
                logger.error(f"Fichier features absent: {self.feature_names_path}")
                return
            with open(self.feature_names_path, "rb") as fn:
                self.feature_names = pickle.load(fn)
                logger.info(f"features chargées via {self.feature_names_path}")
            # Extirpe la liste des noms des features du pkl
            if isinstance(self.feature_names, dict):
                self.feature_names = list(self.feature_names.values())[0]
        # Charge le self des cat et num_features
        self._identify_features()

        # Charge le seuil de validation
        if not self.threshold_path.exists():
            logger.error(f"Fichier seuil absent: {self.threshold_path}")
            return
        with open(self.threshold_path, "rb") as t:
            self.threshold = pickle.load(t)
            logger.info(f"Seuil chargé via {self.threshold_path}")
        # Vérifie et convertie la donnée en float
        if isinstance(self.threshold, np.ndarray):
            self.threshold = self.threshold.item()

    def _identify_features(self):
        """Identifie les colonnes numériques et catégorielles."""
        cat_idx = self.model.get_cat_feature_indices()
        self.cat_features = [self.feature_names[i] for i in cat_idx]
        self.num_features = [
            f for f in self.feature_names if f not in self.cat_features
        ]

    def get_model_info(self) -> dict:
        """
        Compile la métadatas du modèle

        SORTIES:

        model_type: str\n
        n_features: int\n
        feature_names: List[str]
        cat_features: List[str] features catégorielles\n
        num_features: List[str] features numériques\n
        classes: List[str]\n
        threshold: float | None\n
        """

        if self.model is None:
            raise ValueError("Modèle non chargé")

        return {
            "model_type": type(self.model).__name__,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "cat_features": self.cat_features,
            "num_features": self.num_features,
            "classes": self.classes,
            "threshold": self.threshold,
        }

    def predict(self, features: Dict[str, float]) -> Tuple[float, float, str]:
        """
        Réalise une prédiction

        ENTREES:

        features: Dictionnaire des features

        SORTIES:

        Tuple (prediction, confiance, classe)
        """

        if self.model is None:
            raise ValueError("Modèle non chargé")

        # Tf en df
        df = pd.DataFrame([features])

        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"features manquantes:{missing_features}")

        if len(df.columns) != len(self.feature_names):
            raise ValueError("Plus de features qu'attendues")

        # Réarrangement de l'ordre des features suivant l'ordre appris par le modele
        df = df[self.feature_names]

        # Prédiction
        probas = self.model.predict_proba(df)[0]
        if self.threshold is None:
            prediction = int(self.model.predict(df)[0])
            confidence = float(np.max(probas))
        else:
            confidence = float(probas[1])
            prediction = int(confidence >= self.threshold)

        class_name = str(self.classes[prediction])
        prediction_value = float(prediction)

        return prediction_value, confidence, class_name

    def get_feature_importance(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Retourne les top_n features les plus influentes
        """

        if self.model is None:
            raise ValueError("Modèle non chargé")

        if not self.feature_names:
            raise ValueError("Liste des features non définie")

        importances = self.model.get_feature_importance()
        # On convertit explicitement chaque valeur en float Python natif ici
        feature_importance = [
            (name, float(score)) for name, score in zip(self.feature_names, importances)
        ]

        # Tri par valeur absolue décroissante
        top_features = sorted(
            feature_importance, key=lambda item: abs(item[1]), reverse=True
        )

        return top_features[:top_n]
