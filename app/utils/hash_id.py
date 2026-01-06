"""
Empreinte unique et courte.\n
Garantit que les mêmes entrées produiront toujours le même ID,
que ce soit lors d'un import CSV ou d'une requête API en temps réel.
"""

# imports

import hashlib
import json

# ======================


def generate_feature_hash(features: dict) -> str:
    """Génère un ID hashé basé sur le contenu des features."""
    # 1. Trier les clés pour que {"a":1, "b":2} donne le même résultat que {"b":2, "a":1}
    encoded_features = json.dumps(features, sort_keys=True).encode("utf-8")
    # 2. Hasher
    return hashlib.sha256(encoded_features).hexdigest()
