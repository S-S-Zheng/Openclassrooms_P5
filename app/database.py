# C'est le cœur de l'infrastructure de données.
# Il doit être accessible à la fois par les routes et par les scripts de création.
# Configuration de la connexion (Engine, SessionLocal)

import os
from contextlib import contextmanager

from dotenv import load_dotenv

# imports
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

# =================== Mise en place ===========================

#
DATABASE_URL = os.getenv("DATABASE_URL")
# ENGINE: point de départ de SQLAlchemy
# Accepte 3 connexions simultanées et jusqu'à 10 en cas de pic temporaire
# Vérif la connexion toujours valide (indispensable en Cloud)
base_engine = create_engine(
    DATABASE_URL, pool_size=3, max_overflow=7, pool_pre_ping=True
)
# SessionLocal est une factory à sessions pour les routes
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=base_engine)
# Base est la classe mère dont hériteront tous les modèles SQL
Base = declarative_base()


# Fonction utilitaire pour récupérer une session de base de données
@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
