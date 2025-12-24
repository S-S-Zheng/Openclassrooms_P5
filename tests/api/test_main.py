"""
Test de main.
"""

# Imports


# =================== Health =======================
# On s'assure que /health est fonctionnelle: code 200
# On s'assure que la réponse est bien status:ok
def test_healthcheck(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# =================== Root =======================
def test_root(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue sur l'API who's quit"}
