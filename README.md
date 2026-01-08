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
  <a href="https://github.com/S-S-Zheng/Openclassrooms_data_scientist_projects.git">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Projet 5: Déployer un modèle de machine learning</h3>

  <p align="center">
    Projet 5 de la formation d'OpenClassrooms: Data scientist Machine Learning (projet débuté le 08/12/2025)
    <br />
    <a href="https://github.com/S-S-Zheng/Openclassrooms_data_scientist_projects.git"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/S-S-Zheng/Openclassrooms_data_scientist_projects.git">View Demo</a>
    &middot;
    <a href="https://github.com/S-S-Zheng/Openclassrooms_data_scientist_projects.git/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/S-S-Zheng/Openclassrooms_data_scientist_projects.git/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#deployment">Deployment</a></li>
    <li><a href="#authentication">Authentication</a></li>
    <li><a href="#security">Security</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

Dans ce projet, notre objectif est de déployer le modèle ML du projet 4: Classifier automatiquement des informations afin de la rendre accessible via une API.  
Le projet incluera:

1. Un dépôt GitHub contenant le code, l'environnement et la documentation.
2. Une infrastructure CI/CD permettant de garantir la qualité du code, facilitera les tests et permettra un déploiement fiable du modèle.
3. Une API et sa documentation pour exposer le modèle.
4. Une base de donnée PostgreSQL pour intéragir avec le ML.
5. Un ensemble de tests unitaires et fonctionnels servant à garantir la fiabilité et la robustesse du modèle.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

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

### Prerequisites

Le projet nécessite Python 3.11+. Nous utilisons Poetry pour la gestion des dépendances et de l'environnement virtuel.

* Python & Pip

```bash
python --version
# Si besoin d'installation (Debian/Ubuntu)
sudo apt update && sudo apt install python3.11 python3-pip
```

* Poetry (Gestionnaire de dépendances)

```bash
curl -sSL https://install.python-poetry.org | python3 -
# Ajoutez Poetry à votre PATH si nécessaire, puis vérifiez
poetry --version
```

### Installation & Configuration

#### Cloner le Dépôt

```bash
git clone https://github.com/S-S-Zheng/Openclassrooms_P5.git
cd Openclassrooms_P5.git
```

#### Installer l'environnement Vous pouvez choisir entre Poetry (recommandé) ou un environnement virtuel classique

* Via Poetry:

```bash
poetry install
# Activez l'environnement
poetry shell
```

* Via Pip:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
pip install -r requirements-dev.txt
```

#### Variables d'environnement Créez un fichier .env  et env.test à la racine pour configurer l'accès à votre base de données de test local (à créer) et distante Supabase

* Création d'une base PostgreSQL locale:

Installation de psql puis creation depuis le terminale

* .env:

```python
SUPABASE_URL=votre_url_supabase
SUPABASE_KEY=votre_cle_anonyme
DB_PASSWORD=votre_mot_de_passe_db
```

* .env.test:

```python
SUPABASE_URL=votre_url_supabase
SUPABASE_KEY=votre_cle_anonyme
DB_PASSWORD=votre_mot_de_passe_db
```

#### Pour démarrer l'API FastAPI localement avec rechargement automatique (Hot Reload)

```bash
uvicorn app.main:app --reload --port 8000
```

L'API sera disponible sur : [http://localhost:8000](http://localhost:8000). La documentation interactive (Swagger) sera accessible sur : [http://localhost:8000/docs](http://localhost:8000/docs).

#### Avant de pousser vos modifications, vérifiez que l'ensemble de la suite de tests est au vert

```bash
# Lancer tous les tests avec rapport de couverture
pytest tests/
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

L'API expose plusieurs points d'entrée pour interagir avec le modèle de prédiction d'attrition :
Prédiction individuelle : Envoyez un POST sur /predict/ avec les caractéristiques de l'employé.
Analyse du modèle : Accédez à /feature-importance pour comprendre les facteurs clés.
Documentation interactive : Une interface Swagger complète est disponible à la racine.

### Exemple de requête de prédiction via cURL

```text
curl -X 'POST' \
  'https://votre-space.huggingface.co/predict/' \
  -H 'Content-Type: application/json' \
  -d '{
    "features": {
      "age": 35,
      "genre": "m",
      "revenu_mensuel": 4500,
      "frequence_deplacement": "occasionnel",
      "heure_supplementaires": "non"
    }
```

### Documentation Interactive (Swagger)

Une fois l'API lancée, accédez à l'interface Swagger pour tester les endpoints en direct :

* **Local** : `http://localhost:7860/docs`
* **Production** : `https://huggingface.co/spaces/VOTRE_USERNAME/VOTRE_SPACE_NAME/docs`

### Exemple de requête de prédiction

Vous pouvez envoyer une requête POST au format JSON :

```json
{
  "features": {
    "age": 32,
    "genre": "m",
    "revenu_mensuel": 3500,
    "statut_marital": "celibataire",
    "heure_supplementaires": "non"
    // ... remplissez les 21 colonnes
  }
}
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Deployment -->
## Deployment

Le déploiement est entièrement automatisé via une architecture MLOps :
CI (GitHub Actions) : À chaque push, une suite de tests (Unitaire, Fonctionnel, Intégration) est lancée sur un environnement Python 3.11.
Validation des Artefacts : Le pipeline vérifie l'intégrité du modèle CatBoost (.cbm) avant tout déploiement.
CD (GitHub Actions) : Si les tests réussissent sur la branche main, le code est packagé et poussé vers Hugging Face Spaces.
Hébergement Docker : L'API tourne dans un conteneur isolé sur les serveurs de Hugging Face.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Authentication -->
## Authentication

L'accès à la base de données Supabase est sécurisé via des variables d'environnement (secrets) injectées au moment du runtime, garantissant qu'aucun identifiant ne circule en clair dans le dépôt.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Security -->
## Security

La sécurité des données d'entrée est assurée par une double couche de validation :
Validation de Schéma : Pydantic garantit que tous les types de données sont corrects.
Validation métier (Business Logic) : Des règles strictes interdisent les valeurs aberrantes (ex: âge hors [18-65], revenus incohérents).
Protection contre les doublons : Un système de Hash ID unique (SHA-256) pour chaque profil d'entrée évite la redondance dans la base de données et optimise les temps de réponse via un mécanisme de cache.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Strategy of tests -->
## Strategy of tests (a revoir pour son positionnement)

Le projet utilise une pyramide de tests rigoureuse :
Tests Unitaires (tests/unit/) : Vérification des schémas Pydantic et des 21 contraintes métier.
Tests Fonctionnels (tests/functional/) : Validation des routes API, des codes de statut HTTP et des scénarios utilisateurs (Outliers, Missing features).
Tests d'Intégration (tests/integration/) : Vérification de la persistance SQL (Supabase) et de la compatibilité des fichiers binaires du modèle ML.
Pour lancer les tests localement avec couverture :

```bash
pytest --cov=app tests/
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Techncal Doc -->
##  Documentation Technique Complète

La documentation détaillée des modules Python, générée via Sphinx, est disponible dans le dossier docs/_build/html ou consultable sur [GitHub Pages / Lien vers la doc].

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
* [] Documentation Sphinx autogénérée

See the [open issues](https://github.com/github_username/repo_name/issues) for a full list of proposed features (and known issues).

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

<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - <email@email_client.com>

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

<p align="right">(<a href="#readme-top">back to top</a>)</p>

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
