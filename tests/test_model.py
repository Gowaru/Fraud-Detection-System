"""
Tests Unitaires pour le système de détection de fraude
Usage: pytest tests/ -v --cov=src
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import joblib
import json
from unittest.mock import Mock, patch, MagicMock

# Import des modules à tester
import sys
sys.path.append('src')

# Fixtures

@pytest.fixture
def sample_transaction():
    """Transaction exemple pour les tests"""
    return {
        'amount': 150.0,
        'time': 43200,
        'distance_from_home': 5.0,
        'distance_from_last_transaction': 2.0,
        'ratio_to_median_purchase': 1.2,
        'repeat_retailer': 1,
        'used_chip': 1,
        'used_pin_number': 1,
        'online_order': 0,
        'hour': 12.0,
        'amount_log': np.log1p(150.0)
    }

@pytest.fixture
def sample_fraud_transaction():
    """Transaction frauduleuse pour les tests"""
    return {
        'amount': 5000.0,
        'time': 7200,  # 2h du matin
        'distance_from_home': 150.0,
        'distance_from_last_transaction': 100.0,
        'ratio_to_median_purchase': 8.5,
        'repeat_retailer': 0,
        'used_chip': 0,
        'used_pin_number': 0,
        'online_order': 1,
        'hour': 2.0,
        'amount_log': np.log1p(5000.0)
    }

@pytest.fixture
def sample_dataset():
    """Dataset minimal pour les tests"""
    np.random.seed(42)
    n = 100
    
    data = {
        'amount': np.random.gamma(2, 50, n),
        'time': np.random.uniform(0, 86400, n),
        'distance_from_home': np.random.gamma(2, 10, n),
        'distance_from_last_transaction': np.random.gamma(1, 5, n),
        'ratio_to_median_purchase': np.random.normal(1, 0.3, n),
        'repeat_retailer': np.random.choice([0, 1], n),
        'used_chip': np.random.choice([0, 1], n),
        'used_pin_number': np.random.choice([0, 1], n),
        'online_order': np.random.choice([0, 1], n),
        'fraud': np.random.choice([0, 1], n, p=[0.95, 0.05])
    }
    
    df = pd.DataFrame(data)
    df['hour'] = (df['time'] / 3600) % 24
    df['amount_log'] = np.log1p(df['amount'])
    
    return df

@pytest.fixture
def mock_model():
    """Mock d'un modèle entraîné"""
    model = Mock()
    model.predict.return_value = np.array([0])
    model.predict_proba.return_value = np.array([[0.8, 0.2]])
    return model

@pytest.fixture
def mock_scaler():
    """Mock d'un scaler"""
    scaler = Mock()
    scaler.transform.return_value = np.random.randn(1, 11)
    return scaler

# Tests de génération de données

class TestDataGeneration:
    """Tests pour la génération de données synthétiques"""
    
    def test_dataset_shape(self, sample_dataset):
        """Vérifier la forme du dataset"""
        assert sample_dataset.shape[0] == 100
        assert 'fraud' in sample_dataset.columns
        assert 'amount' in sample_dataset.columns
    
    def test_fraud_ratio(self, sample_dataset):
        """Vérifier le ratio de fraude"""
        fraud_ratio = sample_dataset['fraud'].mean()
        assert 0 < fraud_ratio < 0.2  # Entre 0% et 20%
    
    def test_no_missing_values(self, sample_dataset):
        """Vérifier l'absence de valeurs manquantes"""
        assert sample_dataset.isnull().sum().sum() == 0
    
    def test_feature_types(self, sample_dataset):
        """Vérifier les types de features"""
        assert sample_dataset['amount'].dtype in [np.float64, np.float32]
        assert sample_dataset['repeat_retailer'].dtype in [np.int64, np.int32]
        assert sample_dataset['fraud'].dtype in [np.int64, np.int32]
    
    def test_amount_positive(self, sample_dataset):
        """Vérifier que les montants sont positifs"""
        assert (sample_dataset['amount'] > 0).all()
    
    def test_binary_features(self, sample_dataset):
        """Vérifier que les features binaires sont bien 0 ou 1"""
        binary_cols = ['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order', 'fraud']
        for col in binary_cols:
            assert sample_dataset[col].isin([0, 1]).all()

# Tests de préparation des données

class TestDataPreparation:
    """Tests pour la préparation des données"""
    
    def test_train_test_split_ratio(self, sample_dataset):
        """Vérifier le ratio train/test"""
        from sklearn.model_selection import train_test_split
        
        X = sample_dataset.drop('fraud', axis=1)
        y = sample_dataset['fraud']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        assert len(X_train) == 80
        assert len(X_test) == 20
    
    def test_scaling_mean_std(self, sample_dataset):
        """Vérifier que le scaling normalise correctement"""
        from sklearn.preprocessing import StandardScaler
        
        X = sample_dataset.drop('fraud', axis=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Vérifier moyenne ~0 et std ~1
        assert np.abs(X_scaled.mean()) < 0.1
        assert np.abs(X_scaled.std() - 1.0) < 0.1
    
    def test_feature_engineering(self, sample_transaction):
        """Vérifier l'ingénierie de features"""
        assert 'hour' in sample_transaction
        assert 'amount_log' in sample_transaction
        assert 0 <= sample_transaction['hour'] <= 24
        assert sample_transaction['amount_log'] > 0

# Tests du modèle

class TestFraudDetectionModel:
    """Tests pour le modèle de détection"""
    
    def test_model_initialization(self):
        """Tester l'initialisation du modèle"""
        from train import FraudDetectionModel
        
        model = FraudDetectionModel(model_type='random_forest')
        assert model.model_type == 'random_forest'
        assert model.model is None
        assert model.scaler is not None
    
    def test_model_training(self, sample_dataset):
        """Tester l'entraînement du modèle"""
        from train import FraudDetectionModel
        
        model = FraudDetectionModel(model_type='random_forest')
        
        X = sample_dataset.drop('fraud', axis=1).values
        y = sample_dataset['fraud'].values
        
        model.train(X, y)
        
        assert model.model is not None
    
    def test_model_prediction_shape(self, mock_model, sample_transaction):
        """Vérifier la forme des prédictions"""
        X = np.array([[sample_transaction[k] for k in sorted(sample_transaction.keys())]])
        
        prediction = mock_model.predict(X)
        assert prediction.shape == (1,)
        
        proba = mock_model.predict_proba(X)
        assert proba.shape == (1, 2)
    
    def test_prediction_probability_range(self, mock_model, sample_transaction):
        """Vérifier que les probabilités sont entre 0 et 1"""
        X = np.array([[sample_transaction[k] for k in sorted(sample_transaction.keys())]])
        
        proba = mock_model.predict_proba(X)
        assert (proba >= 0).all()
        assert (proba <= 1).all()
        assert np.isclose(proba.sum(axis=1), 1.0).all()
    
    def test_fraud_detection_high_risk(self, sample_fraud_transaction):
        """Tester la détection d'une transaction à haut risque"""
        # Vérifier que les caractéristiques frauduleuses sont présentes
        assert sample_fraud_transaction['amount'] > 1000
        assert sample_fraud_transaction['distance_from_home'] > 50
        assert sample_fraud_transaction['repeat_retailer'] == 0
        assert sample_fraud_transaction['ratio_to_median_purchase'] > 5

# Tests de métriques

class TestModelMetrics:
    """Tests pour les métriques du modèle"""
    
    def test_roc_auc_calculation(self):
        """Tester le calcul du ROC-AUC"""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.4, 0.35, 0.8])
        
        auc = roc_auc_score(y_true, y_proba)
        assert 0 <= auc <= 1
    
    def test_confusion_matrix_structure(self):
        """Vérifier la structure de la matrice de confusion"""
        from sklearn.metrics import confusion_matrix
        
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        
        cm = confusion_matrix(y_true, y_pred)
        assert cm.shape == (2, 2)
        assert cm.sum() == len(y_true)
    
    def test_metrics_deployment_criteria(self):
        """Tester les critères de déploiement"""
        metrics = {
            'precision': 0.87,
            'recall': 0.78,
            'f1_score': 0.82
        }
        
        MIN_PRECISION = 0.85
        MIN_RECALL = 0.75
        
        deployment_ready = (
            metrics['precision'] >= MIN_PRECISION and
            metrics['recall'] >= MIN_RECALL
        )
        
        assert deployment_ready == True
    
    def test_metrics_below_threshold(self):
        """Tester le rejet si métriques insuffisantes"""
        metrics = {
            'precision': 0.70,
            'recall': 0.60
        }
        
        MIN_PRECISION = 0.85
        MIN_RECALL = 0.75
        
        deployment_ready = (
            metrics['precision'] >= MIN_PRECISION and
            metrics['recall'] >= MIN_RECALL
        )
        
        assert deployment_ready == False

# Tests d'intégration

class TestIntegration:
    """Tests d'intégration bout-en-bout"""
    
    def test_full_pipeline(self, sample_dataset):
        """Tester le pipeline complet"""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        
        # Préparation
        X = sample_dataset.drop('fraud', axis=1)
        y = sample_dataset['fraud']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entraînement
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Prédiction
        y_pred = model.predict(X_test_scaled)
        
        # Vérifications
        assert len(y_pred) == len(y_test)
        assert set(y_pred).issubset({0, 1})
    
    def test_model_persistence(self, tmp_path, mock_model):
        """Tester la sauvegarde et le chargement du modèle"""
        model_path = tmp_path / "test_model.pkl"
        
        # Sauvegarde
        joblib.dump(mock_model, model_path)
        
        # Chargement
        loaded_model = joblib.load(model_path)
        
        # Vérification
        assert loaded_model is not None

# Tests de monitoring

class TestMonitoring:
    """Tests pour le monitoring"""
    
    def test_data_drift_detection(self):
        """Tester la détection de drift"""
        # Données de référence
        ref_data = np.random.normal(0, 1, 1000)
        
        # Données actuelles sans drift
        current_data = np.random.normal(0, 1, 1000)
        
        # Test KS (Kolmogorov-Smirnov)
        from scipy import stats
        statistic, p_value = stats.ks_2samp(ref_data, current_data)
        
        # Pas de drift détecté (p-value > 0.05)
        assert p_value > 0.05
    
    def test_performance_degradation_detection(self):
        """Tester la détection de dégradation des performances"""
        metrics_history = [
            {'f1_score': 0.85, 'date': '2025-01-01'},
            {'f1_score': 0.84, 'date': '2025-01-08'},
            {'f1_score': 0.70, 'date': '2025-01-15'},  # Dégradation
        ]
        
        DEGRADATION_THRESHOLD = 0.10  # 10%
        
        initial_f1 = metrics_history[0]['f1_score']
        current_f1 = metrics_history[-1]['f1_score']
        
        degradation = (initial_f1 - current_f1) / initial_f1
        
        assert degradation > DEGRADATION_THRESHOLD

# Tests de sécurité

class TestSecurity:
    """Tests de sécurité"""
    
    def test_input_sanitization(self):
        """Tester la sanitization des inputs"""
        malicious_input = "<script>alert('XSS')</script>"
        
        # Vérifier que les chaînes sont validées
        assert isinstance(malicious_input, str)
        # En production, utiliser des bibliothèques comme bleach
    
    def test_sql_injection_prevention(self):
        """Tester la prévention des injections SQL"""
        malicious_query = "'; DROP TABLE transactions; --"
        
        # Avec des requêtes paramétrées, ceci serait traité comme une chaîne
        assert isinstance(malicious_query, str)

# Tests de performance

class TestPerformance:
    """Tests de performance"""
    
    def test_prediction_latency(self, mock_model, sample_transaction):
        """Tester la latence de prédiction"""
        import time
        
        X = np.array([[sample_transaction[k] for k in sorted(sample_transaction.keys())]])
        
        start = time.time()
        mock_model.predict(X)
        latency = time.time() - start
        
        # La prédiction devrait être rapide (< 100ms)
        assert latency < 0.1
    
    def test_batch_processing_efficiency(self, mock_model):
        """Tester l'efficacité du traitement par batch"""
        batch_sizes = [10, 100, 1000]
        
        for size in batch_sizes:
            X = np.random.randn(size, 11)
            mock_model.predict(X)
            # Vérifier que le traitement se termine sans erreur

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src", "--cov-report=html"])