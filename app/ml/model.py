'''
Charge le modèle catboost optimisé pré-entrainé et gère ses commandes
'''

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import logging # sert a la gestion de logs de python
from catboost import CatBoostClassifier
import shap
import pickle

# Récupère ou crée un logger avec le nom du module courant (ex ici __name__="model")
logger = logging.getLogger(__name__)

# Chemins des fichiers à charger:
model_cbm_path = "model/datas/results/model/best_model.cbm"
features_names_path = "model/datas/results/features_names/features_names.pkl"
threshold_path = "model/datas/results/threshold_opt/thresh_opt.pkl"

class MLModel:
    """Classe pour gérer le modèle ML"""
    
    def __init__(
        self, 
        model_path: str = model_cbm_path,
        features_names_path: str = features_names_path,
        threshold_path: str | None = threshold_path
        ):
        self.model_path = Path(model_path)
        self.features_names_path = Path(features_names_path)
        self.threshold_path = Path(threshold_path)
        
        self.model: CatBoostClassifier | None = None # Instance CBC ou None==> utilité doc
        self.features_names: List[str] | None = None
        self.threshold : float | None = None
        self.classes = ["Employé", "Démissionaire"] # Codé en dur, on verra à changer
        self.explainer = None
        
    def load(self) -> None:
        
        # Charge le modele
        if not self.model_path.exists():
            logger.error(f"Le fichier modèle n'existe pas: {self.model_path}")
            return
        self.model = CatBoostClassifier() # Instantiation de CBC ==> Création de l'objet
        self.model.load_model(self.model_path)
        logger.info(f' Modèle chargé depuis {self.model_path}')
        
        # Charge la liste des features
        if not self.features_names_path.exists():
            logger.error(f"Le fichier des features n'existe pas: {self.features_names_path}")
            return
        with open(self.features_names_path, 'rb') as fn:
            self.features_names  = pickle.load(fn)
            logger.info(f'Liste des features chargée depuis {self.features_names_path}')
        
        # Charge le seuil de validation
        if not self.threshold_path.exists():
            logger.error(f"Le fichier de seuil n'existe pas: {self.threshold_path}")
            return
        with open(self.threshold_path, 'rb') as t:
            self.threshold  = pickle.load(t)
            logger.info(f'Seuil de validation chargé depuis {self.threshold_path}')
        
        # charge SHAP
        self.explainer_global = shap.TreeExplainer(self.model)
        self.explainer_local = shap.TreeExplainer(self.model)
        logger.info('Explainer SHAP chargé')
        
    def predict(
        self, 
        features: List[float]
        ) -> Tuple[float, float, str]:
        '''
        Réalise une prédiction
        
        ENTREES:
        features: Liste des features
        SORTIES:
        Tuple (prediction, confiance, classe)
        '''
        
        if self.model is None:
            raise ValueError("Modèle non chargé")
        
        if self.features_names and len(features) != len(self.features_names):
            raise ValueError(
                f'Nombre de features incorrect: attendu {len(self.features_names)}'
                )
        
        # Reshape de la liste features (1D) en array 2D (1, n_features)(ie tableau numpy)
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
        prediction_value = float(prediction) # conversion float pour compatibilité json
        
        return prediction_value, confidence, class_name
        
    def get_feature_importance(
        self,
        top_n:int = 5
        ) -> List[Tuple[str, float]]:
        '''
        Retourne les top_n features les plus influentes
        '''
        
        if self.model is None:
            raise ValueError("Modèle non chargé")
        
        if not self.features_names:
            raise ValueError("Liste des features non définie")
        
        importances = self.model.get_feature_importance()
        importances_dict = dict(zip(self.features_names, importances))
        
        top_features = sorted(
            importances_dict.items(),
            key=lambda item: abs(item[1]),
            reverse=True
        )
        
        return top_features[:top_n]
        
    def explain_global(
        self,
        features: pd.DataFrame | np.ndarray
        ) -> None:
        '''
        Réalise une beeswarm et scatter plot générale des features
        features : DataFrame ou ndarray (n_samples, n_features)
        '''
        
        if self.model is None:
            raise ValueError("Modèle non chargé")
        
        if not self.features_names:
            raise ValueError("Liste des features non définie")
        
        if isinstance(features, np.ndarray):
            features = pd.DataFrame(features, columns=self.features_names)
        
        
        # Beeswarm
        shap_values_global = self.explainer_global(features)
        
        shap.summary_plot(shap_values_global,features)
        
        for feature in self.features_names:
            shap.plots.scatter(shap_values_global[:, feature],features[feature])
        
    def explain_local(
        self,
        features: List[float] | np.ndarray,
        i: int = 0,
        max_display: int = 5
        ):
        '''
        Réalise une beeswarm et scatter plot générale des features
        features : Liste des features (1D) ou ndarray (1, n_features)
        i : index de l'observation à expliquer
        max_display: nombre max de features a afficher dans le waterfall
        '''
        
        if self.model is None:
            raise ValueError("Modèle non chargé")
        
        if not self.features_names:
            raise ValueError("Liste des features non définie")
        
        # Reshape de la liste features (1D) en array 2D (1, n_features)(ie tableau numpy)
        features = np.array(features).reshape(1, -1)
        
        # Waterfall
        shap_values_local = self.explainer_local(features)
        shap.plots.waterfall(
            shap_values_local[i],
            max_display = max_display
        )
    


# Instance globale
model = MLModel()


