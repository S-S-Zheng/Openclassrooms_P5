"""
Test d'intégration End To End / logique métier
"""

# imports
import pytest

# ======================== test des profiles =========================


@pytest.mark.parametrize(
    "functionnal_profile",
    ["happy_path", "missing_features", "outliers", "over_featured"],
    indirect=True,
)
def test_predict_functionnal(client, functionnal_profile):
    # On récupère le nom du profil utilisé pour ce test précis sinon on attribut 200
    expected_status = functionnal_profile.get("expected_status", 200)
    payload = {"features": functionnal_profile["features"]}

    response = client.post("/predict", json=payload)

    assert (
        response.status_code == expected_status
    ), f"Échec pour le profil {functionnal_profile.get('_profile_name')}: {response.text}"

    # 4. Vérification du contenu (uniquement si le test doit réussir)
    if expected_status == 200:
        json_data = response.json()
        assert "prediction" in json_data
        assert 0 <= json_data["confidence"] <= 1.0
        assert json_data["class_name"] in ["Employé", "Démissionnaire"]
