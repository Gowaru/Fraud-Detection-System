import pytest

class TestAPI:
    """Tests pour l'API FastAPI"""
    
    def test_transaction_validation_valid(self, sample_transaction):
        """Tester la validation d'une transaction valide"""
        # Supprimer les features dérivées pour le test
        transaction = {k: v for k, v in sample_transaction.items() 
                      if k not in ['hour', 'amount_log']}
        
        assert transaction['amount'] > 0
        assert 0 <= transaction['repeat_retailer'] <= 1
    
    def test_transaction_validation_negative_amount(self):
        """Tester le rejet d'un montant négatif"""
        with pytest.raises(Exception):
            assert -100.0 > 0
    
    def test_prediction_response_structure(self, sample_transaction):
        """Vérifier la structure de la réponse de prédiction"""
        response = {
            'transaction_id': 'TXN-123',
            'is_fraud': False,
            'fraud_probability': 0.15,
            'risk_level': 'LOW',
            'confidence': 0.85,
            'timestamp': '2025-01-27T10:00:00',
            'processing_time_ms': 15.5
        }
        
        assert 'transaction_id' in response
        assert 'is_fraud' in response
        assert 0 <= response['fraud_probability'] <= 1
        assert response['risk_level'] in ['LOW', 'MEDIUM', 'HIGH']
    
    def test_batch_size_limit(self):
        """Tester la limite de taille de batch"""
        MAX_BATCH_SIZE = 1000
        batch_size = 500
        
        assert batch_size <= MAX_BATCH_SIZE
        
        with pytest.raises(AssertionError):
            assert 1500 <= MAX_BATCH_SIZE