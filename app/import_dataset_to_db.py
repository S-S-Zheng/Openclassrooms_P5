from datetime import datetime

import pandas as pd
from sqlalchemy.orm import Session  # noqa: F401

from app.api.models_db import Base  # noqa: F401
from app.api.models_db import PredictionRecord
from app.create_db import init_db
from app.database import SessionLocal, engine, get_db  # noqa: F401


def import_csv(file_path: str):
    df = pd.read_csv(file_path)
    import_date = datetime.now().strftime("%d%m%Y")

    # Utilisation clef métier pour distinguer import données et requetes nouvelles
    with get_db() as db:
        print("Importation de données historique ...")
        new_records = []
        for i, (_, row) in enumerate(df.iterrows()):
            # Génération d'un ID unique par ligne : DDMMYYYY + index
            # On s'assure que ça ne dépasse pas 12 caractères
            unique_id = f"{import_date}{i:04d}"

            # Vérification si cet ID existe déjà
            exists = (
                db.query(PredictionRecord)
                .filter(PredictionRecord.id == unique_id)
                .first()
            )
            if exists:
                continue

            # Mapping des classes
            target_value = row.get("a_quitte_l_entreprise")
            if target_value == 0:
                name = "Démissionnaire"
                pred_val = 0
            else:
                name = "Employé"
                pred_val = 1

            record = PredictionRecord(
                id=unique_id,
                inputs=row.to_dict(),
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
