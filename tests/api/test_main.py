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
def test_root_redirects_to_docs(client):
    """Vérifie que la racine redirige vers la documentation."""
    # follow_redirects=False permet de vérifier le code 307 de redirection
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"
