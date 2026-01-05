"""
Permet d'importer un fichier de dataset csv dans la base de données.\n
Vérifie que le dataset n'a jamais été importer dans la base sinon l'ignore.\n
Peut se lancer en tant que tel.
"""

# imports

import pandas as pd

from app.db.create_db import init_db
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

            # Mapping des classes
            if target == 1:
                name = "Démissionnaire"
                pred_val = 1
            else:
                name = "Employé"
                pred_val = 0

            record = PredictionRecord(
                id=unique_id,
                inputs=features,
                prediction=pred_val,
                confidence=1.0,  # données historique donc forcement 1.0
                class_name=name,
            )
            new_records.append(record)

        if new_records:
            db.add_all(new_records)
            db.commit()
            print("Importation réussie.")
        else:
            print("Pas d'importation nécéssaire")


# Empeche le script de se lancer par erreur si appelé par un autre script
if __name__ == "__main__":
    init_db()
    import_csv("hist_datas/Xy.csv")
