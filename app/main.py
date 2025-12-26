"""
Point d'entrée de l'application
"""

from contextlib import asynccontextmanager

# ==================== Imports ==========================
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.api.routes.feature_importance import router as feature_importance_router
from app.api.routes.model_info import router as model_info_router
from app.api.routes.predict import router as predict_router
from app.ml.model import ml_model


# assynccontextmanager est un décorateur qui permet de définir une fonction
# capable de gérer une phase avant de démarrage et une après d'arrêt.
# Ici, tout ce qui est écrit avant yield s'éxé UNE SEULE FOIS au lancement
# du serveur ce qui permet de maintenit l'état tant que le serveur est en ON
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Phase de démarrage: on charge une seule fois les données pour optimiser
    # la RAM
    ml_model.load()
    yield
    # Phase d'arrêt: Nettoyage possible, sauvegarde de données...


# ==================== API =============================
app = FastAPI(
    title="ML Prediction API",
    description="API REST pour exposer le CBC des démissions",
    version="0.1.0",
    lifespan=lifespan,
)

# ==================== Routes ==========================

# include_router permet de centraliser
# le wiring, rend scalable, lisible et testable
app.include_router(predict_router)
app.include_router(feature_importance_router)
app.include_router(model_info_router)

# ==================== ENDPOINTS ========================


# /health
# Test auto CI/CD, debug rapide
# FONDAMENTAL + NE DOIT JAMAIS DEPENDRE DU ML OU DE LA DB
@app.get("/health", tags=["Health"])
def healthcheck():
    """
    Endpoint simple pour vérifier que l'API fonctionne
    """
    return {"status": "ok"}


# / (root)
# Feedback immédiat, debug, UX minimale
@app.get("/", tags=["Root"], include_in_schema=False)
def root():
    """
    Redirige vers la doc Swagger
    """
    return RedirectResponse(url="/docs")
