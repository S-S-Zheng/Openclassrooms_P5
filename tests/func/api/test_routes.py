"""
Suite de tests d'intégration pour les routes de l'API FastAPI.

Ce module valide le comportement des endpoints exposés en simulant des requêtes
HTTP réelles via le TestClient. Il vérifie :
1. Le routage correct des requêtes vers les services (ML et Base de données).
2. La conformité des réponses HTTP (codes de statut et corps de réponse JSON).
3. La gestion des erreurs et la propagation des messages d'exception.
4. L'impact des appels API sur la persistance des données.

L'utilisation de la fixture paramétrée 'error_responses' permet de tester
systématiquement la résilience de chaque route face à des pannes du modèle
ou des erreurs de configuration.
"""

# Imports

from sqlalchemy import delete, func, select

from app.db.models_db import PredictionRecord

# ========================= PREDICT ==========================


# Happy path
def test_predict_success(client, mock_ml_model, func_sample, db_session_for_tests):
    """
    Vérifie le succès complet d'une requête de prédiction.

    S'assure que :
    - La route répond avec un code 200.
    - Les données renvoyées correspondent aux valeurs produites par le modèle.
    - Un enregistrement correspondant est correctement créé en base de données.
    """
    # Nettoyage de la base pour ne pas être influencé par la fixture init
    # permet aussi de ne pas casser le test reset_tables
    db_session_for_tests.execute(delete(PredictionRecord))
    db_session_for_tests.commit()

    mock_predict = mock_ml_model(should_fail=False)
    mock_pred, mock_conf, mock_class = mock_predict.predict.return_value

    response = client.post("/predict/", json=func_sample)

    assert response.status_code == 200

    data = response.json()
    assert data["prediction"] == mock_pred
    assert data["confidence"] == mock_conf
    assert data["class_name"] == mock_class

    record = db_session_for_tests.scalar(select(PredictionRecord))
    assert record is not None
    assert record.prediction == mock_pred
    assert record.confidence == mock_conf
    assert record.class_name == mock_class


# =========================================================


# Erreurs
def test_predict_errors(
    client, mock_ml_model, func_sample, error_responses, db_session_for_tests
):
    """
    Vérifie la gestion des erreurs sur l'endpoint /predict.

    Teste les codes de statut 422, 503 et 500 selon l'état du mock, et confirme
    qu'aucune donnée n'est persistée en cas d'échec de la requête.
    """
    # Nettoyage de la base pour ne pas être influencé par la fixture init
    # permet aussi de ne pas casser le test reset_tables
    db_session_for_tests.execute(delete(PredictionRecord))
    db_session_for_tests.commit()

    mock_ml_model(**error_responses["mock_args"])

    response = client.post("/predict/", json=func_sample)

    assert response.status_code == error_responses["expected_status"]
    assert response.json()["detail"] == error_responses["error_msg"]

    # On s'assure que rien n'est sauvegardé puisque ça plante
    record = db_session_for_tests.scalar(select(func.count(PredictionRecord.id)))
    assert record == 0


# ========================= FEATURE IMPORTANCE ==========================


# Happy path
def test_feature_importance_success(client, mock_ml_model):
    """
    Vérifie l'endpoint d'explicabilité globale.

    Valide que la liste des variables les plus influentes est correctement
    formatée (liste de tuples nom/score) et accessible via une requête GET.
    """
    mock_ml_model(should_fail=False)

    response = client.get("/feature-importance?top_n=5")

    assert response.status_code == 200
    data = response.json()

    assert "top_features" in data
    assert len(data["top_features"]) == 5
    assert isinstance(data["top_features"][0][0], str)
    assert isinstance(data["top_features"][0][1], float)


# =========================================================


# Erreurs
def test_feature_importance_errors(client, mock_ml_model, error_responses):
    """Vérifie la robustesse de la route /feature-importance face aux erreurs du modèle."""
    mock_ml_model(**error_responses["mock_args"])

    response = client.get("/feature-importance?top_n=5")

    assert response.status_code == error_responses["expected_status"]
    assert response.json()["detail"] == error_responses["error_msg"]


# ========================= METADATAS ===========================


# Happy path
def test_model_info_success(client, mock_ml_model):
    """
    Vérifie l'endpoint d'introspection du modèle (/model-info).

    S'assure que toutes les métadonnées techniques (type, noms des features,
    catégorisation, classes et seuil) sont correctement exposées.
    """
    model = mock_ml_model(should_fail=False)

    response = client.get("/model-info")

    assert response.status_code == 200
    data = response.json()

    assert data["model_type"] == "CatBoostClassifier"
    assert data["n_features"] == len(model.feature_names_)
    assert data["feature_names"] == model.feature_names_
    assert data["cat_features"] == ["genre", "statut_marital"]
    assert data["num_features"] == [
        "age",
        "revenu_mensuel",
        "augementation_salaire_precedente",
    ]
    assert data["classes"] == ["Employé", "Démissionnaire"]
    assert data["threshold"] == 0.6


# =========================================================


# Erreurs
def test_model_info_errors(client, mock_ml_model, error_responses):
    """Vérifie la gestion des erreurs lors de la récupération des métadonnées du modèle."""
    mock_ml_model(**error_responses["mock_args"])

    response = client.get("/model-info")

    assert response.status_code == error_responses["expected_status"]
    assert response.json()["detail"] == error_responses["error_msg"]
