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
2. Une infrastructure CI/CD permettant de tester le modèle et de permettre son déploiement rapide et fiable.
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
* [![Gradio][Gradio.com]][Gradio-url]
* [![Pydantic][Pydantic.com]][Pydantic-url]
* [![PostgreSQL][PostgreSQL.com]][PostgreSQL-url]
* [![SQLAlchemy][SQLAlchemy.com]][SQLAlchemy-url]
* [![Pytest-cov][Pytest-cov.com]][Pytest-cov-url]
* [![Swagger][Swagger.com]][Swagger-url]
* [![OpenAPI][OpenAPI.com]][OpenAPI-url]
* [![MkDocs][MkDocs.com]][MkDocs-url]
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

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* python >= 3.13
  ```sh
  # Verifier la version
  python --version
  # Installer python
  sudo apt install python
  # Mettre à jour la version
  sudo apt update python
  sudo apt upgrade python
  ```
* pipx >= 1.7 & poetry >= 2.0
  ```bash
  # Vérifier la version
  pipx --version
  poetry --version
  # Installer pipx et poetry
  pip install pipx
  python -m pipx ensurepath
  # Fermer le terminal et en rouvrir un nouveau
  pipx install poetry
  ```

### Installation

<!-- A suppr peut-être suivant l'avancement
1. Get a free API Key at [https://example.com](https://example.com)
-->
2. Clone the repo
   ```sh
   # Clone le dépôt en local
   git clone https://github.com/S-S-Zheng/Openclassrooms_data_scientist_projects.git
   # Ouvrir le dossier du ML
   cd model
   # Installer l'environnement virtuel et les dépendances via poetry
   poetry install
   # Ou à partir du requirements.txt
   pip install -e .
   ```
<!-- A suppr peut-être suivant l'avancement
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```
5. Change git remote url to avoid accidental pushes to base project
   ```sh
   git remote set-url origin github_username/repo_name
   git remote -v # confirm the changes
   ```
-->
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<span style='color:red;font-weight:bold'> REPRENDRE A PARTIR D ICI (ctrl+shift+v pour preview md vscode)</span>

<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Deployment -->
## Deployment

This part is about how the ML has been deployed?

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Authentication -->
## Authentication

This part is about how one is authenticate?

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Security -->
## Security

This part is about the security

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ROADMAP -->
## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature

See the [open issues](https://github.com/github_username/repo_name/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
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

### Top contributors:

<a href="https://github.com/github_username/repo_name/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=github_username/repo_name" alt="contrib.rocks image" />
</a>



<!-- LICENSE -->
## License

Distributed under the project_license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members

[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers

[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues

[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username

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

[Gradio.com]: https://img.shields.io/badge/Gradio-F97316?logo=Gradio&logoColor=white
[Gradio-url]: https://www.gradio.app/

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

[OpenAPI.com]: https://img.shields.io/badge/OpenAPI-6BA539?logo=openapiinitiative&logoColor=white
[OpenAPI-url]: https://swagger.io/specification/

[MkDocs.com]: https://img.shields.io/badge/MkDocs-526CFE?logo=materialformkdocs&logoColor=fff
[MkDocs-url]: https://www.mkdocs.org/

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


[CatBoost-url]: https://catboost.ai/docs/en/
