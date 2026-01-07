"""
Test fonctionnel sur le vrai modele et les artefacts
"""

# Imports
import pytest

# =========================================================


@pytest.mark.integration
def test_functional_pipeline_real_model(ml_model):
    """Test d'intégration des artefacts."""
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
