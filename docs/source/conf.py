# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
# On remonte de deux niveaux pour atteindre la racine où se trouve le dossier /app
sys.path.insert(0, os.path.abspath('../../'))

project = 'Documentation technique du P5 Openclassrooms'
copyright = '2026, S-S-Zheng'
author = 'S-S-Zheng'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # Génère la doc à partir des docstrings
    'sphinx.ext.napoleon',     # Supporte le format Google/NumPy
    'sphinx.ext.viewcode',     # Ajoute un lien vers le code source
    'sphinx.ext.githubpages',  # Prépare pour GitHub Pages
    'myst_parser',             # Permet de lire le README.md
]

# Configuration pour le Markdown
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'fr'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

# Configuration Autodoc : on veut voir les membres des classes et les fonctions
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': False,
    'inherited-members': False,
    'exclude-members': '__weakref__'
}
# La ligne magique pour stopper les 1000+ erreurs de SQLAlchemy
autodoc_typehints = "none"