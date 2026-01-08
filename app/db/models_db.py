# Pydantic définit la forme des données qui entrent/sortent,
# SQLAlchemy définit la forme des données qui dorment en base.

# imports
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base import Base

# ============== Tables ========================


# Traçabilité
class RequestLog(Base):
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
    a_quitte_l_entreprise = Column(Boolean)

    # Inputs condensé
    inputs = Column(JSONB)

    # Prédiction
    prediction = Column(Integer)
    confidence = Column(Float)
    class_name = Column(String)

    model_version = Column(String, default="v1.0.0")

    logs = relationship("RequestLog", back_populates="prediction_record")
