# C'est le cœur de l'infrastructure de données.
# Il doit être accessible à la fois par les routes et par les scripts de création.
# Configuration de la connexion (Engine, SessionLocal)

import os

# imports
import urllib.parse  # Import indispensable pour les caractères spéciaux
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

# =================== Mise en place ===========================

# ====================== Variables d'environnement ==============
# Récupération sécurisée
db_user = os.getenv("POSTGRES_USER")
db_pass = urllib.parse.quote_plus(
    os.getenv("POSTGRES_PASSWORD", "")
)  # Sécurise le password
db_host = os.getenv("POSTGRES_HOST")
db_port = os.getenv("POSTGRES_PORT")
db_name = os.getenv("POSTGRES_DB")

#
DATABASE_URL = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
# DATABASE_URL = os.getenv("DATABASE_URL")

# ENGINE: point de départ de SQLAlchemy
# Accepte 3 connexions simultanées et jusqu'à 10 en cas de pic temporaire
# Vérif la connexion toujours valide (indispensable en Cloud)
base_engine = create_engine(
    DATABASE_URL, pool_size=3, max_overflow=7, pool_pre_ping=True
)
# SessionLocal est une factory à sessions pour les routes
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=base_engine)


# Fonction utilitaire pour récupérer une session de base de données
def get_db_generator():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# contextmanager est redondant pour FastAPI mais nécéssaire pour les with get_db du coup
# FASTAPI
get_db = get_db_generator

# Pour tests et scripts... on wrappe
get_db_contextmanager = contextmanager(get_db_generator)
