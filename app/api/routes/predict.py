"""
Module de définition du router pour les prédictions d'attrition.

Ce module constitue le point d'entrée principal de l'intelligence artificielle.
Il orchestre le flux de données complet : réception de la requête, journalisation
initiale, vérification de l'existence d'un cache en base de données, exécution
de l'inférence par le modèle CatBoost, persistance du résultat et retour de la
réponse à l'utilisateur.
"""

# ====================== Imports ========================
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.api.schemas import PredictionInput, PredictionOutput
from app.db.actions.get_prediction_from_db import get_prediction
from app.db.actions.save_prediction_to_db import save_prediction
from app.db.database import get_db
from app.utils.logger_db import closing_log, init_log

router = APIRouter(prefix="/predict", tags=["Prediction"])

# ===================== Initialisation du modele =========================


@router.post("/", response_model=PredictionOutput)
def predict(request: Request, payload: PredictionInput, db: Session = Depends(get_db)):
    """
    Réalise une prédiction pour un employé donné.

    Cette méthode suit la pipeline suivante :

    1. **Logging** : Initialisation d'un enregistrement dans 'request_logs'.
    2. **Mise en cache** : Vérification si les caractéristiques ont déjà été traitées
       (via un hash SHA-256) pour retourner un résultat instantané.
    3. **Inférence** : Si nouveau, appel de la méthode predict du modèle chargé.
    4. **Persistance** : Sauvegarde des entrées et de la sortie dans 'predictions'.
    5. **Finalisation** : Calcul du temps de réponse et mise à jour du log.

    Args:
        request (Request): Objet requête FastAPI pour accéder au modèle global.
        payload (PredictionInput): Dictionnaire validé contenant les 21 features.
        db (Session): Session de base de données injectée par dépendance.

    Returns:
        PredictionOutput: Résultat comprenant la prédiction (0/1), le score de
        confiance et le nom de la classe.

    Raises:
        HTTPException: 503 si le modèle n'est pas chargé.
        HTTPException: 422 en cas d'erreur de valeur lors de l'inférence.
        HTTPException: 500 pour les erreurs serveur imprévues.
    """
    # On initialise le temps et le log
    start_time = time.time()
    log_entry = init_log(db, "/predict")

    # Récupération de l'instance du modèle depuis le state de l'application
    model_instance = getattr(request.app.state, "model", None)
    if model_instance is None:
        closing_log(db, log_entry, start_time, status_code=503)
        raise HTTPException(status_code=503, detail="Modèle non chargé sur le serveur")

    # Garde-fou pour empêcher de save plusieurs fois même requete dans db
    cached = get_prediction(db, payload.features)
    if cached:
        closing_log(db, log_entry, start_time, prediction_id=cached.id)
        return PredictionOutput(
            prediction=cached.prediction,
            confidence=cached.confidence,
            class_name=cached.class_name,
        )

    try:
        prediction, confidence, class_name = model_instance.predict(payload.features)
        # Sauvegarde de la requete + ID pour log
        request_id = save_prediction(
            db,
            payload.features,
            (prediction, confidence, class_name),
            log_id=log_entry.id,
        )
        closing_log(db, log_entry, start_time, prediction_id=request_id)
    except ValueError as exc:
        db.rollback()
        closing_log(db, log_entry, start_time, status_code=422)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as e:
        db.rollback()
        closing_log(db, log_entry, start_time, status_code=500)
        raise HTTPException(status_code=500, detail=str(e))

    return PredictionOutput(
        prediction=prediction,
        confidence=confidence,
        class_name=class_name,
    )
