"""
Point d'entrée de l'application
"""

# ==================== Imports ==========================
from fastapi import FastAPI

from app.api.routes.feature_importance import router as feature_importance_router
from app.api.routes.model_info import router as model_info_router
from app.api.routes.predict import router as predict_router

# ==================== API =============================
app = FastAPI(
    title="ML Prediction API",
    description="API REST pour exposer le CBC des démissions",
    version="0.1.0",
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
@app.get("/health", tags=["health"])
def healthcheck():
    """
    Endpoint simple pour vérifier que l'API fonctionne
    """
    return {"status": "ok"}


# / (root)
# Feedback immédiat, debug, UX minimale
@app.get("/", tags=["root"])
def root():
    return {"message": "Bienvenue sur l'API who's quit"}
