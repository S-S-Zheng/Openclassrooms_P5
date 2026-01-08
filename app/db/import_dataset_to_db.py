"""
Module d'importation de données historiques depuis un fichier CSV vers PostgreSQL.

Ce module permet d'étoffer la base de données avec des datasets pré-existants.
Il intègre une logique de dédoublonnage par hachage (SHA-256) pour éviter les
insertions redondantes et assure la traçabilité de l'opération via le système
de logging applicatif.
"""

# imports

import time

import pandas as pd

from app.db.database import get_db_contextmanager
from app.db.models_db import PredictionRecord
from app.utils.hash_id import generate_feature_hash
from app.utils.logger_db import closing_log, init_log

# =============================


def import_csv(file_path: str):
    """
    Lit un fichier CSV et importe les enregistrements uniques dans la table PredictionRecord.

    Le processus suit les étapes suivantes :
    1. Chargement du fichier via Pandas et conversion des NaN en 'None' pour compatibilité JSON.
    2. Initialisation d'un log d'activité pour l'endpoint virtuel '/import'.
    3. Pour chaque ligne : extraction de la target, génération d'un hash ID unique sur les features.
    4. Vérification de l'existence de l'ID en base pour ignorer les doublons.
    5. Construction et insertion massive (bulk insert) des nouveaux enregistrements.

    Args:
        file_path (str): Chemin local vers le fichier CSV contenant les données historiques.

    Raises:
        Exception: En cas d'erreur de lecture, de hachage ou de contrainte d'intégrité SQL,
            une annulation (rollback) est effectuée.

    Note:
        - La 'confidence' est fixée à 1.0 car il s'agit de données historiques observées.
        - Le statut HTTP 201 est loggé en cas de succès avec insertion.
        - Le statut HTTP 204 est loggé si aucun nouvel enregistrement n'a été trouvé.
    """
    start_time = time.time()
    df = pd.read_csv(file_path)
    # Remplacer les NaN par None (car NaN n'est pas un JSON valide)
    df = df.where(pd.notnull(df), None)

    with get_db_contextmanager() as db:
        try:
            # L'endpoint de l'import n'existe pas
            log_entry = init_log(db, "/import")
            print("Importation de données historique ...")

            new_records = []
            for _, row in df.iterrows():
                features = row.to_dict()
                # on retire la target pour ne hasher que les features
                target = features.pop("a_quitte_l_entreprise", None)

                unique_id = generate_feature_hash(features)
                # Vérification si cet ID existe déjà
                if db.get(PredictionRecord, unique_id):
                    continue

                # On constitue un dictionnaire complet qui répond aux exigences de l'UML
                assemble = {
                    "id": unique_id,
                    "inputs": features,
                    "prediction": int(target) if target is not None else None,
                    "confidence": 1.0,  # données historique donc forcement 1.0
                    "class_name": "Démissionnaire" if target == 1 else "Employé",
                    "a_quitte_l_entreprise": (
                        int(target) if target is not None else None
                    ),
                }
                assemble.update(features)

                # On unpack suivant le model UML
                record = PredictionRecord(**assemble)
                new_records.append(record)

            if new_records:
                db.add_all(new_records)
                db.flush()  # permettra d'anticper les erreurs SQL
                closing_log(db, log_entry, start_time, status_code=201)
                print("Importation réussie.")
            else:
                # code 204 = No content
                closing_log(db, log_entry, start_time, status_code=204)
                print("Pas d'importation nécéssaire")

        except Exception as e:
            db.rollback()
            print(f"Erreur lors de l'importation : {e}")
            raise e


if __name__ == "__main__":
    from pathlib import Path

    # On remonte a app/
    BASE_DIR = Path(__file__).resolve().parent.parent
    CSV_PATH = (
        BASE_DIR / "ml" / "model" / "datas" / "results" / "hist_datas" / "P4" / "Xy.csv"
    )

    if not CSV_PATH.exists():
        print("pas d'import depuis le dossier hist_datas")
    else:
        import_csv(CSV_PATH)
