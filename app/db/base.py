"""
On isole Base pour éviter des imports circulaires
"""

# imports
from sqlalchemy.orm import declarative_base

# Base est la classe mère dont hériteront tous les modèles SQL
Base = declarative_base()
