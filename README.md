# 🏦 Fraud Detection System - MLOps Pipeline

## 📋 Description
Système de détection de fraude bancaire avec pipeline MLOps complet, 
orchestré via GitHub Actions et déployé avec Docker.

## 🚀 Fonctionnalités
- Détection de fraude en temps réel
- Pipeline CI/CD automatisé
- Monitoring et alertes
- Versionnement des modèles
- Tests automatisés
- API REST FastAPI
- Dashboards de monitoring

## 📦 Installation

### Prérequis
- Python 3.9+
- Docker & Docker Compose
- Git

### Installation locale
```bash
# Cloner le repo
git clone https://github.com/Gowaru/Fraud-Detection-System.git
cd fraud-detection

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Entraîner le modèle
python src/train.py
```

### Installation avec Docker
```bash
# Build
docker-compose build

# Lancer les services
docker-compose up -d

# Vérifier la santé
curl http://localhost:8000/health
```

## 🎯 Utilisation

### API REST
```python
import requests

# Prédiction simple
transaction = {
    "amount": 150.0,
    "time": 43200,
    "distance_from_home": 5.0,
    "distance_from_last_transaction": 2.0,
    "ratio_to_median_purchase": 1.2,
    "repeat_retailer": 1,
    "used_chip": 1,
    "used_pin_number": 1,
    "online_order": 0
}

response = requests.post(
    "http://localhost:8000/predict",
    json=transaction
)

print(response.json())
```

### CLI
```bash
# visualiser les differentes commandes possibles
python src/train.py --help

# Entraîner le modèle
python src/train.py

# Évaluer le modèle
python scripts/evaluate_model.py

# Détecter le drift
python scripts/detect_drift.py
```

## 🧪 Tests
```bash
# Tests unitaires
pytest tests/ -v

# Avec coverage
pytest tests/ -v --cov=src --cov-report=html

# Tests d'intégration
pytest tests/integration/ -v
```

## 📊 Monitoring
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **API Docs**: http://localhost:8000/docs

## 🔄 CI/CD Pipeline

### Déclenchement
- Push sur main/develop
- Pull request
- Schedule hebdomadaire
- Manuel (workflow_dispatch)

### Étapes
1. Code Quality (lint, format)
2. Tests (unit, integration)
3. Data Validation
4. Model Training
5. Model Evaluation
6. Drift Detection
7. Docker Build
8. Deployment (staging/prod)
9. Monitoring

## 📈 Métriques
- **Precision**: > 85%
- **Recall**: > 75%
- **F1-Score**: > 80%
- **ROC-AUC**: > 0.90

## 🏗️ Architecture
```
fraud-detection/
├── src/
│   └── train.py          # Code d'entraînement
├── tests/
│   ├── test_model.py     # Tests unitaires
│   └── test_api.py       # Tests d'API
├── scripts/
│   ├── validate_data.py  # Validation
│   └── detect_drift.py   # Drift detection
├── models/               # Modèles sauvegardés
├── metrics/              # Métriques
├── api.py               # FastAPI app
├── Dockerfile           # Docker image
├── docker-compose.yml   # Services
└── .github/
    └── workflows/
        └── ci-cd.yml    # Pipeline CI/CD
```

## 📝 License
MIT

## 👥 Contributeurs
- Ansem - Développeur ML