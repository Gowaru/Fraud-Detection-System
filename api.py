"""
API FastAPI pour la Détection de Fraude Bancaire
Expose le modèle ML en production avec monitoring
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import time
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response
import uvicorn

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Métriques Prometheus
PREDICTION_COUNTER = Counter(
    'fraud_predictions_total', 
    'Total number of fraud predictions',
    ['prediction_class']
)
PREDICTION_LATENCY = Histogram(
    'fraud_prediction_latency_seconds',
    'Latency of fraud predictions'
)
API_REQUESTS = Counter(
    'api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status']
)

# Initialisation de l'API
app = FastAPI(
    title="Fraud Detection API",
    description="API ML pour la détection de fraude bancaire en temps réel",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèles Pydantic pour validation
class Transaction(BaseModel):
    """Modèle de transaction bancaire"""
    amount: float = Field(..., gt=0, description="Montant de la transaction en euros")
    time: float = Field(..., ge=0, description="Timestamp en secondes depuis minuit")
    distance_from_home: float = Field(..., ge=0, description="Distance du domicile en km")
    distance_from_last_transaction: float = Field(..., ge=0, description="Distance de la dernière transaction en km")
    ratio_to_median_purchase: float = Field(..., gt=0, description="Ratio par rapport à l'achat médian")
    repeat_retailer: int = Field(..., ge=0, le=1, description="Commerçant habituel (0 ou 1)")
    used_chip: int = Field(..., ge=0, le=1, description="Utilisation de la puce (0 ou 1)")
    used_pin_number: int = Field(..., ge=0, le=1, description="Utilisation du PIN (0 ou 1)")
    online_order: int = Field(..., ge=0, le=1, description="Commande en ligne (0 ou 1)")
    
    @validator('amount')
    def validate_amount(cls, v):
        if v > 100000:
            logger.warning(f"Transaction amount exceptionally high: {v}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "amount": 150.50,
                "time": 43200,
                "distance_from_home": 5.2,
                "distance_from_last_transaction": 2.1,
                "ratio_to_median_purchase": 1.2,
                "repeat_retailer": 1,
                "used_chip": 1,
                "used_pin_number": 1,
                "online_order": 0
            }
        }

class BatchTransactions(BaseModel):
    """Modèle pour traitement par batch"""
    transactions: List[Transaction]
    
    @validator('transactions')
    def validate_batch_size(cls, v):
        if len(v) > 1000:
            raise ValueError("Maximum 1000 transactions per batch")
        return v

class PredictionResponse(BaseModel):
    """Réponse de prédiction"""
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    confidence: float
    timestamp: str
    processing_time_ms: float

class HealthResponse(BaseModel):
    """Réponse health check"""
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    timestamp: str

class ModelInfo(BaseModel):
    """Informations sur le modèle"""
    model_type: str
    version: str
    training_date: str
    metrics: Dict
    features: List[str]

# Variables globales
model = None
scaler = None
model_metadata = {}
start_time = time.time()

def load_model():
    """Chargement du modèle et du scaler"""
    global model, scaler, model_metadata
    
    try:
        logger.info("Loading fraud detection model...")
        model = joblib.load('models/fraud_model_random_forest.pkl')
        scaler = joblib.load('models/scaler.pkl')
        
        # Chargement des métadonnées
        import json
        with open('metrics/metrics.json', 'r') as f:
            model_metadata = json.load(f)
        
        logger.info("✅ Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return False

# Chargement au démarrage
@app.on_event("startup")
async def startup_event():
    """Événement de démarrage"""
    logger.info("🚀 Starting Fraud Detection API...")
    if not load_model():
        logger.error("Failed to load model on startup")
    else:
        logger.info(f"Model metadata: {model_metadata}")

@app.on_event("shutdown")
async def shutdown_event():
    """Événement d'arrêt"""
    logger.info("👋 Shutting down Fraud Detection API...")

# Endpoints

@app.get("/", tags=["General"])
async def root():
    """Endpoint racine"""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint"""
    API_REQUESTS.labels(endpoint='/health', method='GET', status='200').inc()
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=model_metadata.get('timestamp', 'unknown'),
        uptime_seconds=time.time() - start_time,
        timestamp=datetime.now().isoformat()
    )

@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")

@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def model_info():
    """Informations sur le modèle"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    API_REQUESTS.labels(endpoint='/model/info', method='GET', status='200').inc()
    
    return ModelInfo(
        model_type=model_metadata.get('model_type', 'unknown'),
        version=model_metadata.get('timestamp', 'unknown'),
        training_date=model_metadata.get('timestamp', 'unknown'),
        metrics={
            'roc_auc': model_metadata.get('roc_auc', 0),
            'f1_score': model_metadata.get('f1_score', 0),
            'precision': model_metadata.get('precision', 0),
            'recall': model_metadata.get('recall', 0)
        },
        features=[
            'amount', 'time', 'distance_from_home', 
            'distance_from_last_transaction', 'ratio_to_median_purchase',
            'repeat_retailer', 'used_chip', 'used_pin_number', 
            'online_order', 'hour', 'amount_log'
        ]
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_fraud(transaction: Transaction):
    """
    Prédiction de fraude sur une transaction unique
    
    - **amount**: Montant de la transaction
    - **time**: Heure de la transaction (secondes depuis minuit)
    - **distance_from_home**: Distance du domicile
    - **distance_from_last_transaction**: Distance de la dernière transaction
    - **ratio_to_median_purchase**: Ratio par rapport à l'achat médian
    - **repeat_retailer**: Commerçant habituel (0/1)
    - **used_chip**: Puce utilisée (0/1)
    - **used_pin_number**: PIN utilisé (0/1)
    - **online_order**: Commande en ligne (0/1)
    """
    if model is None:
        API_REQUESTS.labels(endpoint='/predict', method='POST', status='503').inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start = time.time()
    
    try:
        # Préparation des features
        transaction_dict = transaction.dict()
        transaction_dict['hour'] = (transaction_dict['time'] / 3600) % 24
        transaction_dict['amount_log'] = np.log1p(transaction_dict['amount'])
        
        # Conversion en DataFrame
        df = pd.DataFrame([transaction_dict])
        
        # Normalisation
        X_scaled = scaler.transform(df)
        
        # Prédiction
        proba = model.predict_proba(X_scaled)[0, 1]
        prediction = int(proba >= 0.5)
        
        # Niveau de risque
        if proba > 0.8:
            risk_level = "HIGH"
        elif proba > 0.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Calcul de la confiance
        confidence = max(proba, 1 - proba)
        
        # Logging
        logger.info(f"Prediction: fraud={prediction}, proba={proba:.4f}, risk={risk_level}")
        
        # Métriques
        PREDICTION_COUNTER.labels(prediction_class='fraud' if prediction else 'legitimate').inc()
        processing_time = (time.time() - start) * 1000
        PREDICTION_LATENCY.observe(processing_time / 1000)
        API_REQUESTS.labels(endpoint='/predict', method='POST', status='200').inc()
        
        # Génération ID unique
        transaction_id = f"TXN-{int(time.time() * 1000)}"
        
        return PredictionResponse(
            transaction_id=transaction_id,
            is_fraud=bool(prediction),
            fraud_probability=float(proba),
            risk_level=risk_level,
            confidence=float(confidence),
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        API_REQUESTS.labels(endpoint='/predict', method='POST', status='500').inc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(batch: BatchTransactions):
    """
    Prédiction de fraude sur un batch de transactions
    Maximum 1000 transactions par requête
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start = time.time()
    results = []
    
    for i, transaction in enumerate(batch.transactions):
        try:
            # Utilisation de l'endpoint single prediction
            result = await predict_fraud(transaction)
            result.transaction_id = f"BATCH-{int(start * 1000)}-{i}"
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing transaction {i}: {e}")
            continue
    
    processing_time = (time.time() - start) * 1000
    
    return {
        "batch_id": f"BATCH-{int(start * 1000)}",
        "total_transactions": len(batch.transactions),
        "successful_predictions": len(results),
        "fraud_detected": sum(1 for r in results if r.is_fraud),
        "total_processing_time_ms": processing_time,
        "results": results
    }

@app.post("/model/reload", tags=["Model"])
async def reload_model():
    """Recharger le modèle (après mise à jour)"""
    logger.info("Reloading model...")
    
    if load_model():
        API_REQUESTS.labels(endpoint='/model/reload', method='POST', status='200').inc()
        return {"status": "success", "message": "Model reloaded successfully"}
    else:
        API_REQUESTS.labels(endpoint='/model/reload', method='POST', status='500').inc()
        raise HTTPException(status_code=500, detail="Failed to reload model")

# Gestion des erreurs
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )