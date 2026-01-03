# Ce script est un utilitaire.
# Le placer dans app/ permet d'importer facilement models_db et database pour créer les tables.

# imports
import sys
from pathlib import Path

from app.api.models_db import PredictionRecord  # noqa: F401
from app.database import Base, base_engine

# Ajout du dossier racine au path pour permettre les imports relatifs
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))


# ====================== Création de la DB ============================


def init_db(reset_tables=False, engine=base_engine):
    if reset_tables:
        Base.metadata.drop_all(bind=engine)
    print("Initialisation de la base de données...")
    try:
        # Crée toutes les tables définies dans models_db qui héritent de Base
        Base.metadata.create_all(bind=engine)
        print("Tables créées avec succès dans PostgreSQL.")
    except Exception as e:
        print(f"Erreur lors de la création de la base : {e}")
        raise e


# Empeche le script de se lancer par erreur si appelé par un autre script
# Inutile à tester puisque c'est juste un déclencheur conditionnel
if __name__ == "__main__":
    init_db()
