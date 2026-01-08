"""
Module de définition des modèles de données ORM (Object-Relational Mapping).

Ce module contient les schémas SQL pour PostgreSQL via SQLAlchemy. Il définit
l'organisation des données stockées, incluant les enregistrements de prédictions
détaillés et le système de journalisation (logging) pour la traçabilité des requêtes.
"""

# Pydantic définit la forme des données qui entrent/sortent,
# SQLAlchemy définit la forme des données qui dorment en base.

# imports
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base import Base

# ============== Tables ========================


# Traçabilité
class RequestLog(Base):
    """
    Modèle représentant les logs d'activité de l'API.

    Stocke les métadonnées de chaque requête entrante pour permettre l'audit de performance
    et la traçabilité des erreurs.
    Chaque log peut être lié à un enregistrement de prédiction spécifique.

    Attributes:
        id (int): Clé primaire auto-incrémentée.
        created_at (datetime): Horodatage de la requête (géré par le serveur SQL).
        endpoint (str): Le point d'entrée API sollicité (ex: '/predict').
        status_code (int): Le code de statut HTTP retourné (ex: 200, 422, 500).
        response_time_ms (float): Temps de traitement de la requête en millisecondes.
        prediction_id (str): Clé étrangère pointant vers l'ID unique de la prédiction associée.
        prediction_record (relationship): Relation ORM vers l'objet PredictionRecord correspondant.
    """

    __tablename__ = "request_logs"

    # Identifications
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Intéractions
    endpoint = Column(String, default="/predict")
    status_code = Column(Integer)
    response_time_ms = Column(Float)

    # Relations
    # Crée une dépendance des ID avec la table predictions (permet la jointure)
    prediction_id = Column(String(64), ForeignKey("predictions.id"))
    # Créée une relation bidirectionnelle entre log et record
    prediction_record = relationship("PredictionRecord", back_populates="logs")


# ===================================================================


# Requete utilisateur
class PredictionRecord(Base):
    """
    Modèle représentant une prédiction stockée et ses caractéristiques d'entrée.

    Cette table contient l'ensemble des variables métier (features) envoyées par l'utilisateur,
    le résultat du modèle ML, et une version sérialisée (JSONB) pour la flexibilité.
    L'ID est un hash SHA-256 des entrées servant de mécanisme de dédoublonnage.

    Attributes:
        id (str): Hash unique des features servant de clé primaire.
        created_at (datetime): Date d'enregistrement.
        age (int): Âge de l'employé.
        genre (str): Genre (m/f).
        revenu_mensuel (int): Revenu mensuel.
        # ... (autres colonnes de caractéristiques métier)
        a_quitte_l_entreprise (int): Valeur réelle observée
        (utilisée pour le réentraînement/historique).
        inputs (JSONB): Copie de sauvegarde de l'intégralité des entrées au format JSON.
        prediction (int): Classe prédite par le modèle (0 ou 1).
        confidence (float): Score de probabilité associé à la prédiction.
        class_name (str): Traduction textuelle de la classe (ex: 'Employé', 'Démissionnaire').
        model_version (str): Version du modèle utilisé lors de l'inférence.
        logs (relationship): Liste des logs de requêtes ayant sollicité cette prédiction précise.
    """

    __tablename__ = "predictions"

    # Identifications
    id = Column(String(64), primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Features
    age = Column(Integer)
    genre = Column(String)
    revenu_mensuel = Column(Integer)
    statut_marital = Column(String)
    poste = Column(String)
    annees_dans_le_poste_actuel = Column(Integer)
    heure_supplementaires = Column(String)
    augementation_salaire_precedente = Column(Integer)
    nombre_participation_pee = Column(Integer)
    nb_formations_suivies = Column(Integer)
    distance_domicile_travail = Column(Integer)
    niveau_education = Column(Integer)
    domaine_etude = Column(String)
    frequence_deplacement = Column(String)
    evolution_note = Column(Integer)
    stagnation_promo = Column(Float)
    freq_chgt_poste = Column(Float)
    revenu_mensuel_ajuste_par_nv_hierarchique = Column(Float)
    revenu_mensuel_par_annee_xp = Column(Float)
    freq_chgt_responsable = Column(Float)
    satisfaction_globale_employee = Column(Integer)

    # Target
    a_quitte_l_entreprise = Column(Integer)

    # Inputs condensé
    inputs = Column(JSONB)

    # Prédiction
    prediction = Column(Integer)
    confidence = Column(Float)
    class_name = Column(String)

    model_version = Column(String, default="v1.0.0")

    logs = relationship("RequestLog", back_populates="prediction_record")
