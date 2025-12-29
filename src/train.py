"""
Système de Détection de Fraude Bancaire avec MLOps
Auteur: Votre Nom
Date: 2025

Usage:
    python train.py                           # Random Forest par défaut
    python train.py --model gradient_boosting # Gradient Boosting
    python train.py --model logistic          # Logistic Regression
    python train.py --model xgboost           # XGBoost
    python train.py --compare                 # Comparer tous les modèles
    python train.py --help                    # Afficher l'aide
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, precision_recall_curve, 
    average_precision_score, f1_score, precision_score, recall_score
)
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("⚠️  imbalanced-learn non installé. Utilisation de class_weight='balanced'")

import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    """Configuration du modèle"""
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    MODEL_PATH = 'models/'
    METRICS_PATH = 'metrics/'
    THRESHOLD = 0.5
    MIN_PRECISION = 0.85  # Seuil minimum pour déploiement
    MIN_RECALL = 0.75

class FraudDetectionModel:
    """
    Modèle de détection de fraude bancaire avec pipeline MLOps
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialisation du modèle
        
        Args:
            model_type (str): Type de modèle ('random_forest', 'gradient_boosting', 'logistic')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.metrics = {}
        self.threshold = Config.THRESHOLD
        
    def create_synthetic_data(self, n_samples=10000):
        """
        Création de données synthétiques pour la démonstration
        Simule des transactions bancaires
        
        Returns:
            pd.DataFrame: Dataset avec transactions
        """
        np.random.seed(Config.RANDOM_STATE)
        
        # Génération de transactions légitimes (95%)
        n_legitimate = int(n_samples * 0.95)
        legitimate = {
            'amount': np.random.gamma(2, 50, n_legitimate),  # Montants normaux
            'time': np.random.uniform(0, 172800, n_legitimate),  # 48h en secondes
            'distance_from_home': np.random.gamma(2, 10, n_legitimate),
            'distance_from_last_transaction': np.random.gamma(1, 5, n_legitimate),
            'ratio_to_median_purchase': np.random.normal(1, 0.3, n_legitimate),
            'repeat_retailer': np.random.choice([0, 1], n_legitimate, p=[0.3, 0.7]),
            'used_chip': np.random.choice([0, 1], n_legitimate, p=[0.1, 0.9]),
            'used_pin_number': np.random.choice([0, 1], n_legitimate, p=[0.2, 0.8]),
            'online_order': np.random.choice([0, 1], n_legitimate, p=[0.6, 0.4]),
            'fraud': np.zeros(n_legitimate)
        }
        
        # Génération de transactions frauduleuses (5%)
        n_fraud = n_samples - n_legitimate
        fraud = {
            'amount': np.random.gamma(5, 100, n_fraud),  # Montants élevés
            'time': np.random.uniform(0, 172800, n_fraud),
            'distance_from_home': np.random.gamma(5, 30, n_fraud),  # Loin de chez soi
            'distance_from_last_transaction': np.random.gamma(4, 20, n_fraud),
            'ratio_to_median_purchase': np.random.normal(3, 1, n_fraud),  # Ratio élevé
            'repeat_retailer': np.random.choice([0, 1], n_fraud, p=[0.8, 0.2]),  # Nouveau commerçant
            'used_chip': np.random.choice([0, 1], n_fraud, p=[0.6, 0.4]),  # Moins de puce
            'used_pin_number': np.random.choice([0, 1], n_fraud, p=[0.7, 0.3]),  # Moins de PIN
            'online_order': np.random.choice([0, 1], n_fraud, p=[0.3, 0.7]),  # Plus en ligne
            'fraud': np.ones(n_fraud)
        }
        
        # Combinaison
        df_legitimate = pd.DataFrame(legitimate)
        df_fraud = pd.DataFrame(fraud)
        df = pd.concat([df_legitimate, df_fraud], ignore_index=True)
        
        # Mélange
        df = df.sample(frac=1, random_state=Config.RANDOM_STATE).reset_index(drop=True)
        
        # Ajout d'features dérivées
        df['hour'] = (df['time'] / 3600) % 24
        df['amount_log'] = np.log1p(df['amount'])
        
        return df
    
    def prepare_data(self, df):
        """
        Préparation des données
        
        Args:
            df (pd.DataFrame): Dataset brut
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        print("📊 Préparation des données...")
        
        # Séparation features et target
        X = df.drop('fraud', axis=1)
        y = df['fraud']
        
        self.feature_names = X.columns.tolist()
        
        # Split train/test stratifié
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=Config.TEST_SIZE, 
            random_state=Config.RANDOM_STATE,
            stratify=y
        )
        
        # Normalisation
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✅ Données préparées: {X_train.shape[0]} train, {X_test.shape[0]} test")
        print(f"   Fraudes dans train: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
        print(f"   Fraudes dans test: {y_test.sum()} ({y_test.mean()*100:.2f}%)")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_model(self):
        """
        Sélection du modèle avec gestion du déséquilibre
        
        Returns:
            Model ou Pipeline: Modèle configuré
        """
        if self.model_type == 'random_forest':
            classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=4,
                class_weight='balanced',
                random_state=Config.RANDOM_STATE,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            classifier = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=Config.RANDOM_STATE
            )
        elif self.model_type == 'logistic':
            classifier = LogisticRegression(
                class_weight='balanced',
                random_state=Config.RANDOM_STATE,
                max_iter=1000,
                solver='lbfgs'
            )
        elif self.model_type == 'xgboost':
            try:
                import xgboost as xgb
                classifier = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=Config.RANDOM_STATE,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
                print("✅ XGBoost chargé avec succès")
            except ImportError:
                print("⚠️  XGBoost non installé (pip install xgboost)")
                print("   Utilisation de Gradient Boosting à la place...")
                classifier = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=Config.RANDOM_STATE
                )
                self.model_type = 'gradient_boosting'
        else:
            raise ValueError(
                f"Modèle '{self.model_type}' non supporté. "
                f"Choisir parmi: random_forest, gradient_boosting, logistic, xgboost"
            )
        
        # Pipeline avec SMOTE si disponible
        if IMBLEARN_AVAILABLE and self.model_type != 'logistic':
            pipeline = ImbPipeline([
                ('smote', SMOTE(random_state=Config.RANDOM_STATE, k_neighbors=3)),
                ('classifier', classifier)
            ])
            print("✅ Pipeline avec SMOTE activé")
            return pipeline
        else:
            # Sinon, retourner juste le classifier avec class_weight
            print("✅ Utilisation de class_weight='balanced'")
            return classifier
    
    def train(self, X_train, y_train):
        """
        Entraînement du modèle avec validation croisée
        
        Args:
            X_train: Features d'entraînement
            y_train: Labels d'entraînement
        """
        print(f"\n🚀 Entraînement du modèle {self.model_type}...")
        
        self.model = self.get_model()
        
        # Validation croisée
        cv = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='f1')
        
        print(f"   CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Entraînement final
        self.model.fit(X_train, y_train)
        
        print("✅ Modèle entraîné avec succès")
    
    def evaluate(self, X_test, y_test):
        """
        Évaluation complète du modèle
        
        Args:
            X_test: Features de test
            y_test: Labels de test
            
        Returns:
            dict: Métriques de performance
        """
        print("\n📈 Évaluation du modèle...")
        
        # Prédictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calcul des métriques
        self.metrics = {
            'timestamp': datetime.now().isoformat(),
            'model_type': self.model_type,
            'accuracy': float(np.mean(y_pred == y_test)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred)),
            'roc_auc': float(roc_auc_score(y_test, y_proba)),
            'average_precision': float(average_precision_score(y_test, y_proba)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Rapport détaillé
        print("\n" + "="*60)
        print(classification_report(y_test, y_pred, target_names=['Légitime', 'Fraude']))
        print("="*60)
        
        print(f"\n🎯 Métriques clés:")
        print(f"   Precision: {self.metrics['precision']:.4f}")
        print(f"   Recall: {self.metrics['recall']:.4f}")
        print(f"   F1-Score: {self.metrics['f1_score']:.4f}")
        print(f"   ROC-AUC Score: {self.metrics['roc_auc']:.4f}")
        
        # Vérification des seuils de déploiement
        self.metrics['deployment_ready'] = (
            self.metrics['precision'] >= Config.MIN_PRECISION and 
            self.metrics['recall'] >= Config.MIN_RECALL
        )
        
        if self.metrics['deployment_ready']:
            print(f"\n✅ MODÈLE PRÊT POUR DÉPLOIEMENT")
        else:
            print(f"\n⚠️  MODÈLE NON PRÊT - Amélioration nécessaire")
            print(f"   Precision: {self.metrics['precision']:.4f} (min: {Config.MIN_PRECISION})")
            print(f"   Recall: {self.metrics['recall']:.4f} (min: {Config.MIN_RECALL})")
        
        return self.metrics
    
    def plot_results(self, X_test, y_test):
        """
        Visualisation des résultats
        
        Args:
            X_test: Features de test
            y_test: Labels de test
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Matrice de confusion
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title(f'Matrice de Confusion - {self.model_type}')
        axes[0, 0].set_ylabel('Réel')
        axes[0, 0].set_xlabel('Prédit')
        
        # 2. Distribution des scores de prédiction
        y_proba = self.model.predict_proba(X_test)[:, 1]
        axes[0, 1].hist(y_proba[y_test == 0], bins=50, alpha=0.5, label='Légitime', color='green')
        axes[0, 1].hist(y_proba[y_test == 1], bins=50, alpha=0.5, label='Fraude', color='red')
        axes[0, 1].axvline(self.threshold, color='black', linestyle='--', label='Seuil')
        axes[0, 1].set_xlabel('Score de Probabilité')
        axes[0, 1].set_ylabel('Fréquence')
        axes[0, 1].set_title('Distribution des Scores')
        axes[0, 1].legend()
        
        # 3. Courbe Precision-Recall
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        axes[1, 0].plot(recall, precision, linewidth=2)
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title(f'Precision-Recall Curve (AP={self.metrics["average_precision"]:.3f})')
        axes[1, 0].grid(True)
        
        # 4. Feature Importance (si disponible)
        if IMBLEARN_AVAILABLE and hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
            importances = self.model.named_steps['classifier'].feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            
            axes[1, 1].barh(range(10), importances[indices])
            axes[1, 1].set_yticks(range(10))
            axes[1, 1].set_yticklabels([self.feature_names[i] for i in indices])
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].set_title('Top 10 Features Importantes')
            axes[1, 1].invert_yaxis()
        elif not IMBLEARN_AVAILABLE and hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            
            axes[1, 1].barh(range(10), importances[indices])
            axes[1, 1].set_yticks(range(10))
            axes[1, 1].set_yticklabels([self.feature_names[i] for i in indices])
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].set_title('Top 10 Features Importantes')
            axes[1, 1].invert_yaxis()
        else:
            axes[1, 1].text(0.5, 0.5, f'Feature Importance\nnon disponible pour\n{self.model_type}',
                          ha='center', va='center', fontsize=14)
            axes[1, 1].set_title('Feature Importance')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Créer le dossier si nécessaire
        import os
        os.makedirs(Config.METRICS_PATH, exist_ok=True)
        
        filename = f'{Config.METRICS_PATH}evaluation_{self.model_type}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 Graphiques sauvegardés: {filename}")
        plt.close()
    
    def save_model(self):
        """Sauvegarde du modèle et des artefacts"""
        import os
        os.makedirs(Config.MODEL_PATH, exist_ok=True)
        os.makedirs(Config.METRICS_PATH, exist_ok=True)
        
        # Sauvegarde du modèle avec le nom du type
        model_filename = f"{Config.MODEL_PATH}fraud_model_{self.model_type}.pkl"
        joblib.dump(self.model, model_filename)
        
        # Sauvegarde du scaler
        scaler_filename = f"{Config.MODEL_PATH}scaler_{self.model_type}.pkl"
        joblib.dump(self.scaler, scaler_filename)
        
        # Sauvegarde des métriques
        metrics_filename = f"{Config.METRICS_PATH}metrics_{self.model_type}.json"
        with open(metrics_filename, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        print(f"\n💾 Modèle sauvegardé: {model_filename}")
        print(f"💾 Scaler sauvegardé: {scaler_filename}")
        print(f"💾 Métriques sauvegardées: {metrics_filename}")
    
    def predict(self, transaction):
        """
        Prédiction sur une nouvelle transaction
        
        Args:
            transaction (dict): Dictionnaire avec les features
            
        Returns:
            dict: Résultat de la prédiction
        """
        # Conversion en DataFrame
        df = pd.DataFrame([transaction])
        
        # Normalisation
        X_scaled = self.scaler.transform(df)
        
        # Prédiction
        proba = self.model.predict_proba(X_scaled)[0, 1]
        prediction = int(proba >= self.threshold)
        
        return {
            'fraud_probability': float(proba),
            'is_fraud': bool(prediction),
            'risk_level': 'HIGH' if proba > 0.8 else 'MEDIUM' if proba > 0.5 else 'LOW'
        }
    
    @staticmethod
    def create_transaction(amount, hour_of_day, distance_from_home, 
                          distance_from_last_transaction, ratio_to_median_purchase,
                          repeat_retailer, used_chip, used_pin_number, online_order):
        """
        Créer une transaction avec calcul automatique des features dérivées
        
        Args:
            amount (float): Montant en euros
            hour_of_day (float): Heure de la journée (0-24)
            distance_from_home (float): Distance du domicile en km
            distance_from_last_transaction (float): Distance de la dernière transaction en km
            ratio_to_median_purchase (float): Ratio par rapport à l'achat médian
            repeat_retailer (int): Commerçant habituel (0 ou 1)
            used_chip (int): Puce utilisée (0 ou 1)
            used_pin_number (int): PIN utilisé (0 ou 1)
            online_order (int): Commande en ligne (0 ou 1)
            
        Returns:
            dict: Transaction complète avec features dérivées
        """
        # Calculer time en secondes depuis minuit
        time_seconds = hour_of_day * 3600
        
        return {
            'amount': float(amount),
            'time': float(time_seconds),
            'distance_from_home': float(distance_from_home),
            'distance_from_last_transaction': float(distance_from_last_transaction),
            'ratio_to_median_purchase': float(ratio_to_median_purchase),
            'repeat_retailer': int(repeat_retailer),
            'used_chip': int(used_chip),
            'used_pin_number': int(used_pin_number),
            'online_order': int(online_order),
            'hour': float(hour_of_day),  # Stocké directement
            'amount_log': float(np.log1p(amount))
        }


def parse_arguments():
    """Parser les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(
        description='Système de Détection de Fraude Bancaire - MLOps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python train.py                           # Random Forest (défaut)
  python train.py --model gradient_boosting # Gradient Boosting
  python train.py --model logistic          # Logistic Regression
  python train.py --model xgboost           # XGBoost
  python train.py --compare                 # Comparer tous les modèles
  python train.py --samples 50000           # 50k échantillons
  python train.py --model random_forest --samples 20000 --no-viz
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='random_forest',
        choices=['random_forest', 'gradient_boosting', 'logistic', 'xgboost'],
        help='Type de modèle à entraîner (défaut: random_forest)'
    )
    
    parser.add_argument(
        '--samples', '-s',
        type=int,
        default=10000,
        help='Nombre d\'échantillons à générer (défaut: 10000)'
    )
    
    parser.add_argument(
        '--compare', '-c',
        action='store_true',
        help='Comparer tous les modèles disponibles'
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Désactiver la génération de graphiques'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Ne pas sauvegarder le modèle'
    )
    
    return parser.parse_args()


def train_single_model(model_type, n_samples=10000, generate_viz=True, save_model=True):
    """
    Entraîner un seul modèle
    
    Args:
        model_type (str): Type de modèle
        n_samples (int): Nombre d'échantillons
        generate_viz (bool): Générer les visualisations
        save_model (bool): Sauvegarder le modèle
    
    Returns:
        dict: Métriques du modèle
    """
    print("="*70)
    print(f"🏦 DÉTECTION DE FRAUDE BANCAIRE - Modèle: {model_type.upper()}")
    print("="*70)
    
    # Initialisation
    model = FraudDetectionModel(model_type=model_type)
    
    # Création de données synthétiques
    print(f"\n📦 Génération de {n_samples} transactions synthétiques...")
    df = model.create_synthetic_data(n_samples=n_samples)
    
    # Préparation
    X_train, X_test, y_train, y_test = model.prepare_data(df)
    
    # Entraînement
    model.train(X_train, y_train)
    
    # Évaluation
    metrics = model.evaluate(X_test, y_test)
    
    # Visualisation
    if generate_viz:
        model.plot_results(X_test, y_test)
    
    # Sauvegarde
    if save_model:
        model.save_model()
    
    # Tests de prédiction avec plusieurs scénarios
    print("\n" + "="*70)
    print("🧪 TESTS DE PRÉDICTION SUR DIFFÉRENTS SCÉNARIOS")
    print("="*70)
    
    # Scénario 1: Transaction FRAUDULEUSE (suspicieuse)
    print("\n" + "─"*70)
    print("🚨 SCÉNARIO 1: Transaction Suspecte")
    print("─"*70)
    
    fraud_transaction = FraudDetectionModel.create_transaction(
        amount=5000.0,
        hour_of_day=2.0,  # 2h du matin
        distance_from_home=150.0,
        distance_from_last_transaction=100.0,
        ratio_to_median_purchase=8.5,
        repeat_retailer=0,
        used_chip=0,
        used_pin_number=0,
        online_order=1
    )
    
    result = model.predict(fraud_transaction)
    print(f"📋 Détails:")
    print(f"   💰 Montant: {fraud_transaction['amount']:.2f}€ (très élevé)")
    print(f"   🕐 Heure: {fraud_transaction['hour']:.1f}h (nuit)")
    print(f"   📍 Distance domicile: {fraud_transaction['distance_from_home']:.0f}km (très loin)")
    print(f"   📊 Ratio achat médian: {fraud_transaction['ratio_to_median_purchase']:.1f}x (anormal)")
    print(f"   🏪 Commerçant habituel: {'Non' if fraud_transaction['repeat_retailer'] == 0 else 'Oui'} (nouveau)")
    print(f"   💳 Puce utilisée: {'Non' if fraud_transaction['used_chip'] == 0 else 'Oui'}")
    print(f"   🔢 PIN utilisé: {'Non' if fraud_transaction['used_pin_number'] == 0 else 'Oui'}")
    print(f"\n🎯 Résultat de détection:")
    print(f"   Probabilité de fraude: {result['fraud_probability']:.2%}")
    print(f"   Verdict: {'🚨 FRAUDE DÉTECTÉE' if result['is_fraud'] else '✅ Transaction légitime'}")
    print(f"   Niveau de risque: {result['risk_level']}")
    
    # Scénario 2: Transaction LÉGITIME (normale)
    print("\n" + "─"*70)
    print("✅ SCÉNARIO 2: Transaction Légitime")
    print("─"*70)
    
    legit_transaction = FraudDetectionModel.create_transaction(
        amount=85.50,
        hour_of_day=12.0,  # Midi
        distance_from_home=3.2,
        distance_from_last_transaction=1.5,
        ratio_to_median_purchase=0.9,
        repeat_retailer=1,
        used_chip=1,
        used_pin_number=1,
        online_order=0
    )
    
    result = model.predict(legit_transaction)
    print(f"📋 Détails:")
    print(f"   💰 Montant: {legit_transaction['amount']:.2f}€ (normal)")
    print(f"   🕐 Heure: {legit_transaction['hour']:.1f}h (jour)")
    print(f"   📍 Distance domicile: {legit_transaction['distance_from_home']:.1f}km (proche)")
    print(f"   📊 Ratio achat médian: {legit_transaction['ratio_to_median_purchase']:.1f}x (normal)")
    print(f"   🏪 Commerçant habituel: {'Oui' if legit_transaction['repeat_retailer'] == 1 else 'Non'} (connu)")
    print(f"   💳 Puce utilisée: {'Oui' if legit_transaction['used_chip'] == 1 else 'Non'}")
    print(f"   🔢 PIN utilisé: {'Oui' if legit_transaction['used_pin_number'] == 1 else 'Non'}")
    print(f"\n🎯 Résultat de détection:")
    print(f"   Probabilité de fraude: {result['fraud_probability']:.2%}")
    print(f"   Verdict: {'🚨 FRAUDE DÉTECTÉE' if result['is_fraud'] else '✅ Transaction légitime'}")
    print(f"   Niveau de risque: {result['risk_level']}")
    
    # Scénario 3: Transaction LIMITE (cas ambigu)
    print("\n" + "─"*70)
    print("⚠️  SCÉNARIO 3: Transaction Limite (cas ambigu)")
    print("─"*70)
    
    limit_transaction = FraudDetectionModel.create_transaction(
        amount=850.0,
        hour_of_day=21.0,  # 21h (soir)
        distance_from_home=25.0,
        distance_from_last_transaction=15.0,
        ratio_to_median_purchase=2.5,
        repeat_retailer=0,
        used_chip=1,
        used_pin_number=1,
        online_order=1
    )
    
    result = model.predict(limit_transaction)
    print(f"📋 Détails:")
    print(f"   💰 Montant: {limit_transaction['amount']:.2f}€ (élevé)")
    print(f"   🕐 Heure: {limit_transaction['hour']:.1f}h (soir)")
    print(f"   📍 Distance domicile: {limit_transaction['distance_from_home']:.0f}km (moyen)")
    print(f"   📊 Ratio achat médian: {limit_transaction['ratio_to_median_purchase']:.1f}x (inhabituel)")
    print(f"   🏪 Commerçant habituel: {'Non' if limit_transaction['repeat_retailer'] == 0 else 'Oui'} (nouveau)")
    print(f"   💳 Puce utilisée: {'Oui' if limit_transaction['used_chip'] == 1 else 'Non'}")
    print(f"   🔢 PIN utilisé: {'Oui' if limit_transaction['used_pin_number'] == 1 else 'Non'}")
    print(f"\n🎯 Résultat de détection:")
    print(f"   Probabilité de fraude: {result['fraud_probability']:.2%}")
    print(f"   Verdict: {'🚨 FRAUDE DÉTECTÉE' if result['is_fraud'] else '✅ Transaction légitime'}")
    print(f"   Niveau de risque: {result['risk_level']}")
    
    return metrics


def compare_all_models(n_samples=10000):
    """
    Comparer tous les modèles disponibles
    
    Args:
        n_samples (int): Nombre d'échantillons
    """
    print("="*70)
    print("🏆 COMPARAISON DE TOUS LES MODÈLES")
    print("="*70)
    
    models_to_test = ['random_forest', 'gradient_boosting', 'logistic', 'xgboost']
    results = {}
    
    # Générer les données une seule fois
    print(f"\n📦 Génération de {n_samples} transactions synthétiques...")
    temp_model = FraudDetectionModel(model_type='random_forest')
    df = temp_model.create_synthetic_data(n_samples=n_samples)
    X_train, X_test, y_train, y_test = temp_model.prepare_data(df)
    
    # Tester chaque modèle
    for model_type in models_to_test:
        print(f"\n{'='*70}")
        print(f"🧪 Test du modèle: {model_type.upper()}")
        print(f"{'='*70}")
        
        try:
            model = FraudDetectionModel(model_type=model_type)
            model.scaler = temp_model.scaler  # Réutiliser le même scaler
            model.feature_names = temp_model.feature_names
            
            model.train(X_train, y_train)
            metrics = model.evaluate(X_test, y_test)
            model.plot_results(X_test, y_test)
            model.save_model()
            
            results[model_type] = metrics
            
        except Exception as e:
            print(f"❌ Erreur avec {model_type}: {e}")
            continue
    
    # Afficher le comparatif
    print("\n" + "="*70)
    print("📊 TABLEAU COMPARATIF DES MODÈLES")
    print("="*70)
    print(f"{'Modèle':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
    print("-"*70)
    
    for model_type, metrics in results.items():
        print(f"{model_type:<20} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} "
              f"{metrics['f1_score']:<12.4f} "
              f"{metrics['roc_auc']:<12.4f}")
    
    # Identifier le meilleur modèle
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
        print("\n" + "="*70)
        print(f"🏆 MEILLEUR MODÈLE: {best_model[0].upper()}")
        print(f"   F1-Score: {best_model[1]['f1_score']:.4f}")
        print(f"   Precision: {best_model[1]['precision']:.4f}")
        print(f"   Recall: {best_model[1]['recall']:.4f}")
        print(f"   ROC-AUC: {best_model[1]['roc_auc']:.4f}")
        
        if best_model[1]['deployment_ready']:
            print(f"   ✅ Prêt pour déploiement")
        else:
            print(f"   ⚠️  Amélioration nécessaire")
        print("="*70)
    
    return results


def main():
    """Pipeline principal avec arguments"""
    
    # Parser les arguments
    args = parse_arguments()
    
    # Créer les dossiers nécessaires
    os.makedirs(Config.MODEL_PATH, exist_ok=True)
    os.makedirs(Config.METRICS_PATH, exist_ok=True)
    
    try:
        if args.compare:
            # Mode comparaison
            compare_all_models(n_samples=args.samples)
        else:
            # Mode entraînement simple
            train_single_model(
                model_type=args.model,
                n_samples=args.samples,
                generate_viz=not args.no_viz,
                save_model=not args.no_save
            )
        
        print("\n" + "="*70)
        print("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()