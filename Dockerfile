# Dockerfile nécessaire pour utiliser FastAPI avec HF
FROM python:3.11-slim

# Installation des dépendances système pour CatBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Création de l'utilisateur (sys pas postgres)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

WORKDIR /app

# Copie et installation des dépendances
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copie du code et des poids du modèle
COPY --chown=user . .

# Exposition du port (FastAPI par défaut)
EXPOSE 7860

# Commande de lancement (adaptée à Hugging Face Spaces)
CMD python -m app.db.import_dataset_to_db && \
    uvicorn app.main:app --host 0.0.0.0 --port 7860