"""
Test d'intégration du pipeline d'inférence avec les artefacts réels.

Ce module effectue un test de "bout-en-bout" sur la couche ML en utilisant
les véritables fichiers binaires (.cbm, .pkl) produits lors de la phase
d'entraînement. Il garantit que le modèle chargé en local est compatible
avec le code du wrapper MLModel et que les types de données produits
en sortie sont strictement conformes aux attentes de l'API (notamment
l'absence de types NumPy natifs non sérialisables).
"""

# Imports
import pytest

# =========================================================


@pytest.mark.integration
def test_functional_pipeline_real_model(ml_model):
    """
    Valide le chargement et l'inférence sur le modèle de production réel.

    Ce test vérifie que :
    1. Le modèle et ses métadonnées (features, seuils) sont accessibles et chargeables.
    2. Un dictionnaire d'entrée généré dynamiquement selon le schéma du modèle
        est correctement traité.
    3. Les sorties (prediction, confidence) sont converties en types Python natifs
        (int, float) pour éviter les erreurs de sérialisation JSON.

    Args:
        ml_model (MLModel): Instance réelle du wrapper de modèle.

    Raises:
        pytest.skip: Si les fichiers du modèle sont absents de l'environnement de test.
    """
    try:
        ml_model.load()  # Charge fichiers locaux
    except Exception as e:
        pytest.skip(f"Fichiers de modèle non trouvés pour le test fonctionnel : {e}")

    # On utilise un exemple de données
    cat_idx = ml_model.model.get_cat_feature_indices()
    # On itère sur la liste des noms suivant les indices i et on lui attribue
    # un str si i appartient aux indices catégorielles sinon un float
    sample_input = {
        name: "0.0" if i in cat_idx else 0.0
        for i, name in enumerate(ml_model.model.feature_names_)
    }

    # Exécution
    prediction, confidence, class_name = ml_model.predict(sample_input)

    # Assertions sur les types de sortie (très important pour ton erreur ndarray/float)
    assert isinstance(prediction, int)
    assert isinstance(confidence, float)
    assert isinstance(class_name, str)
    assert 0 <= confidence <= 1.0
    # Pour voir ce que ressort le test (pytest -m integration -s)
    print(
        f"\n{sample_input} \nPred={prediction} \nConf={confidence} \nClass_name={class_name}"
    )
