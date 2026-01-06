"""
Permet d'importer un fichier de dataset csv dans la base de données.\n
Vérifie que le dataset n'a jamais été importer dans la base sinon l'ignore.\n
Peut se lancer en tant que tel.
"""

# imports

import pandas as pd

from app.db.database import get_db_contextmanager
from app.db.models_db import PredictionRecord
from app.utils.hash_id import generate_feature_hash

# =============================


def import_csv(file_path: str):
    df = pd.read_csv(file_path)
    # Remplacer les NaN par None (car NaN n'est pas un JSON valide)
    df = df.where(pd.notnull(df), None)

    with get_db_contextmanager() as db:
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
                "a_quitte_l_entreprise": int(target) if target is not None else None,
            }
            assemble.update(features)

            # On unpack suivant le model UML
            record = PredictionRecord(**assemble)
            new_records.append(record)

        if new_records:
            db.add_all(new_records)
            db.commit()
            print("Importation réussie.")
        else:
            print("Pas d'importation nécéssaire")


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
