# Projet 5: Déployer un modèle de Machine Learning
<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!-- A décommenter si besoin dans le futur
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![project_license][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
-->

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/S-S-Zheng/Openclassrooms_P5.git">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Projet 5: Déployer un modèle de machine learning</h3>

  <p align="center">
    Projet 5 de la formation d'OpenClassrooms: Data scientist Machine Learning (projet débuté le 08/12/2025)
    <br />
    <a href="https://github.com/S-S-Zheng/Openclassrooms_P5.git"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/S-S-Zheng/Openclassrooms_P5.git">View Demo</a>
    &middot;
    <a href="https://github.com/S-S-Zheng/Openclassrooms_P5.git/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/S-S-Zheng/Openclassrooms_P5.git/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Sommaire</summary>
  <ol>
    <li>
      <a href="#about-the-project">A propos du projet</a>
      <ul>
        <li><a href="#built-with">Construit avec</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Débuter</a>
      <ul>
        <li><a href="#prerequisites">Prerequis</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#deployment">Deploiement</a></li>
    <li><a href="#authentication">Authentification</a></li>
    <li><a href="#security">Securité</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#technical_details">Documentation technique</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## A propos du projet

[![Product Name Screen Shot][product-screenshot]](https://example.com)

Dans ce projet, notre objectif est de déployer le modèle ML du projet 4: Classifier automatiquement des informations afin de la rendre accessible via une API.  
Le projet incluera:

1. Un dépôt GitHub contenant le code, l'environnement et la documentation.
2. Une infrastructure CI/CD permettant de garantir la qualité du code, facilitera les tests et permettra un déploiement fiable du modèle.
3. Une API et sa documentation pour exposer le modèle.
4. Une base de donnée PostgreSQL pour intéragir avec le ML.
5. Un ensemble de tests unitaires et fonctionnels servant à garantir la fiabilité et la robustesse du modèle.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Architecture du projet

```text
livrable_P5
├── app/                                      # Cœur de l'application (FastAPI)
│   ├── api/                                  # Couche interface
│   │   ├── routes/                           # Points d'entrée (Endpoints) de l'API
│   │   │   ├── feature_importance.py         # /feature-importance/
│   │   │   ├── model_info.py                 # /model-info/
│   │   │   └── predict.py                    # /predict/
│   │   └── schemas.py                        # Validation des données (Pydantic)
│   ├── db/                                   # Couche de persistance (SQLAlchemy/PostgreSQL)
│   │   ├── actions/                          # Intéractions limitées avec la DB
│   │   │   ├── get_prediction_from_db.py     # Intérroge la DB
│   │   │   └── save_prediction_to_db.py      # Enregistre les prédictions et log
│   │   ├── base.py                           # Déclaration de la base de données
│   │   ├── create_db.py                      # Script d'initialisation des tables
│   │   ├── database.py                       # Configuration de la session Engine
│   │   ├── import_dataset_to_db.py           # Pipeline d'importation CSV vers SQL
│   │   └── models_db.py                      # Schémas des tables SQL
│   ├── ml/                                   # Couche Intelligence Artificielle
│   │   ├── model/                            
│   │   │   ├── datas/                        # Données, artefacts et modèle ML entrainé
│   │   └── model.py                          # Classe wrapper du modèle CatBoostClassifier
│   ├── utils/                                # Utilitaires transverses
│   │   ├── hash_id.py                        # Génération des IDs anonymisés (SHA-256)
│   │   └── logger_db.py                      # Système de log personnalisé vers la DB
│   └── main.py                               # Point d'entrée principal de l'API
├── images/                                   # Images pour la doc README.md
├── tests/                                    # Suite de tests automatisée (Pytest)
│   ├── fixtures/                             # Exemple de requêtes fonctionnelles (YAML/JSON)
│   ├── func/                                 # Tests fonctionnels (E2E & Routes)
│   ├── integration/                          # Tests de communication avec la base de données
│   └── unit/                                 # Tests unitaires des composants isolés
├── coverage.ini                              # Configuration du rapport de couverture
├── Dockerfile                                # Instructions de conteneurisation (Hugging Face)
├── init_db.sh                                # Script Shell pour initialiser la DB locale
├── LICENSE                                   # Licence MIT du projet
├── pytest.ini                                # Configuration globale de l'env de test
├── README.md                                 # Documentation principale du projet
├── requirements-dev.txt                      # Dépendances complètes (Dev/Test/Lint)
└── requirements.txt                          # Dépendances minimales pour la production
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Construit avec

Le projet 5:

* [![GitHub][GitHub.com]][GitHub-url]
* [![Hugging Face Spaces][Hugging Face Spaces.com]][Hugging Face Spaces-url]
* [![GitHub Actions][GitHub Actions.com]][GitHub Actions-url]
* [![FastAPI][FastAPI.com]][FastAPI-url]
* [![Pydantic][Pydantic.com]][Pydantic-url]
* [![PostgreSQL][PostgreSQL.com]][PostgreSQL-url]
* [![SQLAlchemy][SQLAlchemy.com]][SQLAlchemy-url]
* [![Pytest-cov][Pytest-cov.com]][Pytest-cov-url]
* [![Swagger][Swagger.com]][Swagger-url]
* [![Sphinx][Sphinx.com]][Sphinx-url]

Le modèle du projet 4:

* [![Python][Python.com]][Python-url]
* [![Poetry][Poetry.com]][Poetry-url]
* [![Jupyter][Jupyter.com]][Jupyter-url]
* [![Matplotlib][Matplotlib.com]][Matplotlib-url]
* [![NumPy][NumPy.com]][NumPy-url]
* [![Pandas][Pandas.com]][Pandas-url]
* [![Scikit-learn][Scikit-learn.com]][Scikit-learn-url]
* [![CatBoost][CatBoost.com]][CatBoost-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

Suivez ces instructions pour installer le projet localement et lancer l'API sur votre machine de développement.

### Prerequis

#### Local

Le projet nécessite Python 3.11+ et d'une database locale. Il est possible de rester sur Pip ou d'utiliser Poetry pour la gestion des dépendances.

* **Pip ou Poetry**

```bash
# Vérifier la version python
python --version

# Si besoin d'installation (Debian/Ubuntu)
sudo apt update && sudo apt install python3.11 python3-pip

# Ou Poetry
# curl -sSL https://install.python-poetry.org | python3 -
# # Ajoutez Poetry à votre PATH si nécessaire, puis vérifiez
# poetry --version
```

* **Création d'une base PostgreSQL locale**

  Pour exécuter les tests d'intégration, vous devez disposer d'une instance PostgreSQL locale.

  1. *Installation* :

      ```bash
      sudo apt install postgresql postgresql-contrib
      ```

  2. *Accès au terminal psql* :

      ```bash
      sudo -u postgres psql
      ```

  3. *Initialisation de la DB* :

      ```sql
      CREATE DATABASE ml_test_db;
      CREATE USER test_user WITH PASSWORD 'votre_mot_de_passe';
      GRANT ALL PRIVILEGES ON DATABASE ml_test_db TO test_user;
      ```

#### Distant

* **Création d'un dépôt distant GitHub** :

    Créez un compte sur [GitHub](https://github.com/), créez un nouveau dépôt vide et connectez votre projet local :

    ```bash
    git remote add origin https://github.com/votre-user/votre-projet.git
    git push -u origin main
    ```

    Pour les secrets, allez dans Settings > Secrets and variables > Actions pour ajouter vos secrets (HUGGINGFACE_TOKEN, SB_HOST, etc.).

* **Création d'un espace sur Hugging Face**:

    Créez un compte sur Hugging Face, cliquez sur "New Space", choisissez le SDK Docker et un nom pour votre projet.Une fois l'espace créé, dans vos paramètres de profil, créez un "Write Token" pour permettre à GitHub de pousser le code.

    Concernant les secrets, allez dans les paramètres de votre Space, ajoutez les variables d'environnement de votre base de données Supabase pour que l'API puisse s'y connecter au runtime.

* **Création d'une base PostgreSQL Supabase**:

    Créez un compte et un projet sur Supabase puis cliquez sur le bouton Connect sur la barre de tâche supérieure à côté du nom de la base pour récupérez les informations de connexion. A noté que vous pourrez reset votre mot de passe de la base si celui-ci ne vous convient plus dans Project Settings > Database.

* **Création d'une base PostgreSQL locale**

  Pour exécuter les tests d'intégration, vous devez disposer d'une instance PostgreSQL locale.

  1. *Installation* :

      ```bash
      sudo apt install postgresql postgresql-contrib
      ```

  2. *Accès au terminal psql* :

      ```bash
      sudo -u postgres psql
      ```

  3. *Initialisation de la DB* :

      ```sql
      CREATE DATABASE ml_test_db;
      CREATE USER test_user WITH PASSWORD 'votre_mot_de_passe';
      GRANT ALL PRIVILEGES ON DATABASE ml_test_db TO test_user;
      ```

#### Distant

* **Création du dépôt GitHub, de l'interface Hugging Face et de la database Supabase**

  1. *Création d'un dépôt distant GitHub* :

      Créez un compte sur [GitHub](https://github.com/), créez un nouveau dépôt vide et connectez votre projet local :

      ```bash
      git remote add origin https://github.com/votre-user/votre-projet.git
      git push -u origin main
      ```

      Pour les secrets, allez dans Settings > Secrets and variables > Actions pour ajouter vos secrets (HUGGINGFACE_TOKEN, SB_HOST, etc.).

  2. *Création d'un espace sur Hugging Face*:

      Créez un compte sur Hugging Face, cliquez sur "New Space", choisissez le SDK Docker et un nom pour votre projet.Une fois l'espace créé, dans vos paramètres de profil, créez un "Write Token" pour permettre à GitHub de pousser le code.

      Concernant les secrets, allez dans les paramètres de votre Space, ajoutez les variables d'environnement de votre base de données Supabase pour que l'API puisse s'y connecter au runtime.

  3. *Création d'une base PostgreSQL Supabase*:

      Créez un compte et un projet sur Supabase puis cliquez sur le bouton Connect sur la barre de tâche supérieure à côté du nom de la base pour récupérez les informations de connexion. A noté que vous pourrez reset votre mot de passe de la base si celui-ci ne vous convient plus dans Project Settings > Database.

### Installation & Configuration

#### Cloner le Dépôt

```bash
git clone https://github.com/S-S-Zheng/Openclassrooms_P5.git
cd Openclassrooms_P5.git
```

#### Installer l'environnement

* **Pip ou Poetry**:

  ```bash
    # Activer l'environnement virtuel
    python -m venv venv
    source venv/bin/activate

    # Pip
    pip install -r requirements-dev.txt

    # # Poetry
    # poetry init
    # poetry add -r requirements-dev.txt
    # poetry install
    # poetry shell
    ```

#### Variables d'environnement

Créez un fichier .env et env.test à la racine pour configurer l'accès à L'espace de Hugging Face, votre base de données de test local (à créer) et distante Supabase. **TOUJOURS VERIFIER LEUR PRESENCE DANS .gitignore. CES FICHIERS NE DOIVENT JAMAIS APPARAITRE DANS LE DEPOT**

1. **.env**:

    ```python
    HUGGINGFACE_USERNAME = Nom_utilisateur_HF
    HUGGINGFACE_SPACE_NAME = Nom_espace_HF
    HUGGINGFACE_TOKEN = Token_HF

    SB_USER=Nom_utilisateur_SB
    SB_PASSWORD=Mot_de_passe_database_SB
    SB_HOST=Nom_hote_SB
    SB_PORT=Numero_port_SB
    SB_DB=Nom_database_SB
    ```

2. **.env.test**:

    ```python
    POSTGRES_USER=Nom_utilisateur_local
    POSTGRES_PASSWORD=Mot_de_passe_database_loca
    POSTGRES_HOST=Nom_hote_local
    POSTGRES_PORT=Numero_port_local
    POSTGRES_DB=Nom_database_local
    ```

#### Test local

Pour démarrer l'API FastAPI localement avec rechargement automatique (Hot Reload):

  ```bash
  uvicorn app.main:app --reload --port 7860
  ```

L'API sera disponible sur : [http://localhost:7860](http://localhost:7860).

Avant de pousser vos modifications, vérifiez que l'ensemble de la suite de tests est au vert:

  ```bash
  # Lancer tous les tests avec rapport de couverture
  pytest tests/
  ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

L'API expose plusieurs points d'entrée pour interagir avec le modèle de prédiction :

* **Prédiction individuelle** :
  Envoyez un POST sur /predict/ avec les caractéristiques de l'employé.
* **Analyse des features** :
  Accédez à /feature-importance pour comprendre les facteurs clefs de la démission.
* **Informations sur le modèle** :
  Accédez à /model-info/ pour avoir des informations sur le modèle.
* **Documentation interactive** :
  Une interface Swagger complète est disponible à la racine.

### Exemple de requête de prédiction via cURL

```text
curl -X 'POST' \
  'https://s254-p5-oc.hf.space/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "features": {
    "age": 41,
    "annees_dans_le_poste_actuel": 4,
    "augementation_salaire_precedente": 11,
    "distance_domicile_travail": 1,
    "domaine_etude": "infra & cloud",
    "evolution_note": 0,
    "freq_chgt_poste": 0.888889,
    "freq_chgt_responsable": 0.833333,
    "frequence_deplacement": "occasionnel",
    "genre": "f",
    "heure_supplementaires": "oui",
    "nb_formations_suivies": 0,
    "niveau_education": 2,
    "nombre_participation_pee": 0,
    "poste": "cadre commercial",
    "revenu_mensuel": 5993,
    "revenu_mensuel_ajuste_par_nv_hierarchique": 2996.5,
    "revenu_mensuel_par_annee_xp": 665.888889,
    "satisfaction_globale_employee": 8,
    "stagnation_promo": 0,
    "statut_marital": "célibataire"
  }
}'
```

### Documentation Interactive (Swagger)

Une fois l'API lancée, accédez à l'interface Swagger pour tester les endpoints en direct :

* **Local** :
  `http://localhost:7860/docs`
* **Production** :
  `https://huggingface.co/spaces/Nom_utilisateur_HF/Nom_espace_HF`

### Exemple de requête de prédiction

Vous pouvez envoyer une requête POST au format JSON :

```json
{
  "features": {
    "age": 41,
    "annees_dans_le_poste_actuel": 4,
    "augementation_salaire_precedente": 11,
    "distance_domicile_travail": 1,
    "domaine_etude": "infra & cloud",
    "evolution_note": 0,
    "freq_chgt_poste": 0.888889,
    "freq_chgt_responsable": 0.833333,
    "frequence_deplacement": "occasionnel",
    "genre": "f",
    "heure_supplementaires": "oui",
    "nb_formations_suivies": 0,
    "niveau_education": 2,
    "nombre_participation_pee": 0,
    "poste": "cadre commercial",
    "revenu_mensuel": 5993,
    "revenu_mensuel_ajuste_par_nv_hierarchique": 2996.5,
    "revenu_mensuel_par_annee_xp": 665.888889,
    "satisfaction_globale_employee": 8,
    "stagnation_promo": 0,
    "statut_marital": "célibataire"
  }
}
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Deployment -->
## Deploiement

Le déploiement est entièrement automatisé via une architecture MLOps :

* Hook pré-commit (filtre local):
  Le hook de pré-commit s'exécute automatiquement sur ta machine à chaque tentative de git commit. Son rôle est de s'assurer qu'aucun code "sale" ou mal formaté ne quitte ton poste de travail.

* CI (GitHub Actions) :
  Chaque modification poussée sur les branches main ou develop déclenche automatiquement le pipeline.
  Il est constitué de deux jobs:
  
  lint : Vérifie la conformité du code en passant par isort, black et flake8 (bien que le pré-commit le fasse déjà, la présence de ce job permet de s'assurer que le linting est effectuée)

  test : Ne se lance que si le linting est validé, ce job a pour fonction de tester le comportement unitaire et fonctionnel du code. Il lance un conteneur éphemère PostgreSQL 15 de test et execute pytest en suivant les directives du pytest.ini.

* CD (GitHub Actions) :
  Le déploiement ne se lance que si le pipeline de CI a réussi (workflow_run success). Il est restreint à la branche main pour garantir que seul le code de production est déployé.

  Au lieu de pousser tout le dépôt (ce qui serait lourd et risqué), le script crée un dossier éphémère hf_deploy. Il sélectionne les fichiers afin de garantir une conteneurisation optimisée; La seléction comprend le code de l'API, la logique ML et les artefacts associés (.cbm, .pkl). Cela garantit que les secrets locaux ou les données volumineuses (notebooks, png, csv ...) ne sont jamais exposés sur Hugging Face.

  Le script génère aussi dynamiquement un README.md avec un bloc YAML (frontmatter). C'est ce fichier qui configure Hugging Face (SDK Docker, port 7860, version Python, licence).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Authentication -->
## Authentification

L'accès à la base de données Supabase est sécurisé via des variables d'environnement unitaires (secrets) injectées au moment du runtime, garantissant qu'aucun identifiant ne circule en clair dans le dépôt. Les variables sont gérées via les GitHub Secrets et Hugging Face Secrets, isolant les secrets même pendant l'exécution du conteneur.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Security -->
## Securité

La sécurité des données d'entrée est assurée par une plusieurs couches de validation :

### Protection des Données et de la DB

* **Proxy de Données** :
  L'utilisateur n'interagit jamais directement avec la base de données ; l'API agit comme une passerelle sécurisée.
* **Connexion Chiffrée & Pooler** :
Utilisation d'un connection pooler avec chiffrement SSL pour toutes les communications avec Supabase.
* **Protection contre les Doublons** :
  Système de Hash unique (SHA-256) pour chaque prédiction, garantissant l'intégrité et servant de mécanisme de cache sécurisé.
* **Traçabilité (Auditing)** :
  Journalisation systématique de chaque requête dans une table de logs dédiée pour l'audit et la détection d'anomalies.

### Validation et Intégrité

* **Validation de Schéma (Pydantic)** :
  Filtrage strict des types de données en entrée de l'API.
* **Contrôle de Cohérence Métier** :
  Rejet automatique des données ne respectant pas la logique métier (ex: revenus anormaux [1200-100000], âge hors limites [18-65], heure supplémentaire en booleen, etc.).

### Sécurité du Runtime (Hugging Face)

* **Gestion des Secrets** :
  Utilisation des "Repository Secrets" de GitHub et des "Space Secrets" de Hugging Face. Aucun identifiant n'est présent dans le code.
* **Isolation Docker** :
  Exécution dans un environnement sandboxé sans privilèges root.
* **Filtrage des Sorties** :
  Les exceptions internes sont interceptées pour ne jamais exposer la structure de la base de données ou du système dans les réponses HTTP.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Technical Doc -->
##  Documentation Technique Complète

La documentation détaillée des modules Python est générée via Sphinx et est disponible dans le dossier docs/build/html et générable via:

  ```bash
  # Générer la documentation Sphinx
  cd docs
  make html
  ```

Un pdf "documentation_technique" est fourni avec dans docs/.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

* [x] Création d'un dépôt Github pour le projet
* [x] Création d'une classe qui regroupe les méthodes de ML
* [x] Mise en place de tests - part 1
* [x] Automatisation CI/CD (GitHub Actions) - part 1
* [x] Création de l'API avec FastAPI
* [x] Mise en place de tests - part 2
* [x] Automatisation CI/CD (GitHub Actions) - part 2
* [x] Mise en place de la base de données PostgreSQL (Supabase)
* [x] Mise en place de tests - part 3
* [x] Automatisation CI/CD (GitHub Actions) - part 3
* [x] Documentation Sphinx autogénérée

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Top contributors

<a href="https://github.com/github_username/repo_name/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=github_username/repo_name" alt="contrib.rocks image" />
</a>
-->

<!-- LICENSE -->
## License

Distributed under the project_license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - <email@email_client.com>

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

ACKNOWLEDGMENTS
## Acknowledgments

<p align="right">(<a href="#readme-top">back to top</a>)</p>
-->

<!-- =============================================================================== -->
<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[product-screenshot]: images/Gemini_P5.png

<!-- Shields.io badges. You can a comprehensive list with many more badges at: https://github.com/inttter/md-badges -->
<!--
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/

[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/

[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/

[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/

[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/

[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com

[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com

[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com

[Gradio.com]: https://img.shields.io/badge/Gradio-F97316?logo=Gradio&logoColor=white
[Gradio-url]: https://www.gradio.app/

[OpenAPI.com]: https://img.shields.io/badge/OpenAPI-6BA539?logo=openapiinitiative&logoColor=white
[OpenAPI-url]: https://swagger.io/specification/

[MkDocs.com]: https://img.shields.io/badge/MkDocs-526CFE?logo=materialformkdocs&logoColor=fff
[MkDocs-url]: https://www.mkdocs.org/
-->

<!--My list of badges-->
[GitHub.com]: https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white
[GitHub-url]: https://github.com/

[Hugging Face Spaces.com]: https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000
[Hugging Face Spaces-url]: https://huggingface.co/spaces

[GitHub Actions.com]: https://img.shields.io/badge/GitHub_Actions-2088FF?logo=github-actions&logoColor=white
[GitHub Actions-url]: https://github.com/features/actions

[FastAPI.com]: https://img.shields.io/badge/FastAPI-009485.svg?logo=fastapi&logoColor=white
[FastAPI-url]: https://fastapi.tiangolo.com/

[Pydantic.com]: https://img.shields.io/badge/Pydantic-E92063?logo=Pydantic&logoColor=white
[Pydantic-url]: https://docs.pydantic.dev/latest/

[PostgreSQL.com]: https://img.shields.io/badge/Postgres-%23316192.svg?logo=postgresql&logoColor=white
[PostgreSQL-url]: https://www.postgresql.org/

[SQLAlchemy.com]: https://img.shields.io/badge/SQLALCHEMY-D71F00?style=for-the-badge&logo=sqlalchemy&logoColor=white&logoSize=auto
[SQLAlchemy-url]: https://www.sqlalchemy.org/

[Pytest-cov.com]: https://img.shields.io/badge/Pytest--cov-%233F51B5?style=for-the-badge&logo=pytest&logoColor=white&labelColor=black
[Pytest-cov-url]: https://pypi.org/project/pytest-cov/

[Swagger.com]: https://img.shields.io/badge/Swagger-85EA2D?logo=swagger&logoColor=173647
[Swagger-url]: https://swagger.io/

[Sphinx.com]: https://img.shields.io/badge/Sphinx-000?logo=sphinx&logoColor=fff
[Sphinx-url]: https://www.sphinx-doc.org/en/master/

<!--Utilisés pour le modèle-->
[Python.com]: https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff
[Python-url]: https://www.python.org/

[Poetry.com]: https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json
[Poetry-url]: https://python-poetry.org/

[Jupyter.com]: https://img.shields.io/badge/Jupyter-ffffff?logo=JupyterB
[Jupyter-url]: https://jupyter.org/

[Matplotlib.com]: https://custom-icon-badges.demolab.com/badge/Matplotlib-71D291?logo=matplotlib&logoColor=fff
[Matplotlib-url]: https://matplotlib.org/

[NumPy.com]: https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff
[NumPy-url]: https://numpy.org/

[Pandas.com]: https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=fff
[Pandas-url]: https://pandas.pydata.org/docs/index.html

[Scikit-learn.com]: https://img.shields.io/badge/-scikit--learn-%23F7931E?logo=scikit-learn&logoColor=white
[Scikit-learn-url]: https://scikit-learn.org/stable/index.html

[CatBoost.com]: https://img.shields.io/badge/CatBoost-FF4632?logo=catboost&logoColor=white
[CatBoost-url]: https://catboost.ai/docs/en/
