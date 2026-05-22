import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, 
                             confusion_matrix, 
                             roc_auc_score,
                             precision_score,
                             recall_score,
                             f1_score)
from imblearn.over_sampling import SMOTE
import os
import pickle

class FraudDetectionModels:
    def __init__(self):
        self.logistic_model = None
        self.isolation_model = None
        self.models_dir = 'data'
    def split_data(self, X, y):
        """Sépare les données en ensemble d'entraînement et de test"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42,
            stratify=y
        )
        
        print(f"✅ Données séparées :")
        print(f"   Entraînement : {len(X_train)} transactions")
        print(f"   Test : {len(X_test)} transactions")
        print(f"   Fraudes dans test : {y_test.sum()}")
        
        return X_train, X_test, y_train, y_test
    def apply_smote(self, X_train, y_train):
        """Rééquilibre les données avec SMOTE"""
        print("⚙️ Application de SMOTE...")
        
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"✅ Données rééquilibrées :")
        print(f"   Transactions normales : {sum(y_resampled==0)}")
        print(f"   Transactions frauduleuses : {sum(y_resampled==1)}")
        
        return X_resampled, y_resampled
    def train_logistic(self, X_train, y_train):
        """Entraîne la Logistic Regression"""
        print("🔄 Entraînement Logistic Regression...")
        
        self.logistic_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        self.logistic_model.fit(X_train, y_train)
        
        # Sauvegarder le modèle
        filepath = os.path.join(self.models_dir, 'logistic_model.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(self.logistic_model, f)
        
        print("✅ Logistic Regression entraînée et sauvegardée")
        return self.logistic_model
    def train_isolation_forest(self, X_train):
        """Entraîne l'Isolation Forest"""
        print(" Entraînement Isolation Forest...")
        
        self.isolation_model = IsolationForest(
            contamination=0.002,
            random_state=42,
            n_estimators=100
        )
        self.isolation_model.fit(X_train)
        
        # Sauvegarder le modèle
        filepath = os.path.join(self.models_dir, 'isolation_model.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(self.isolation_model, f)
        
        print("✅ Isolation Forest entraîné et sauvegardé")
        return self.isolation_model
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Évalue les performances d'un modèle"""
        if model_name == 'Isolation Forest':
            y_pred = model.predict(X_test)
            y_pred = np.where(y_pred == -1, 1, 0)
        else:
            y_pred = model.predict(X_test)
        
        metrics = {
            'model_name': model_name,
            'precision': round(precision_score(y_test, y_pred), 4),
            'recall': round(recall_score(y_test, y_pred), 4),
            'f1_score': round(f1_score(y_test, y_pred), 4),
            'roc_auc': round(roc_auc_score(y_test, y_pred), 4),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        print(f"\n Résultats {model_name} :")
        print(f"   Précision : {metrics['precision']}")
        print(f"   Recall : {metrics['recall']}")
        print(f"   F1-Score : {metrics['f1_score']}")
        print(f"   ROC-AUC : {metrics['roc_auc']}")
        
        return metrics
    def predict_transaction(self, transaction_data):
        """Prédit si une transaction est frauduleuse"""
        results = {}
        
        if self.logistic_model:
            proba = self.logistic_model.predict_proba(
                transaction_data)[0][1]
            results['logistic'] = {
                'prediction': 'FRAUD' if proba > 0.5 else 'NORMAL',
                'confidence': round(proba * 100, 2)
            }
        
        if self.isolation_model:
            iso_pred = self.isolation_model.predict(transaction_data)[0]
            results['isolation_forest'] = {
                'prediction': 'FRAUD' if iso_pred == -1 else 'NORMAL',
                'anomaly_score': round(
                    self.isolation_model.score_samples(transaction_data)[0], 4)
            }
        
        return results
    def load_models(self):
        """Recharge les modèles déjà entraînés"""
        logistic_path = os.path.join(self.models_dir, 'logistic_model.pkl')
        isolation_path = os.path.join(self.models_dir, 'isolation_model.pkl')
        
        if os.path.exists(logistic_path):
            with open(logistic_path, 'rb') as f:
                self.logistic_model = pickle.load(f)
            print("✅ Logistic Regression chargée")
        
        if os.path.exists(isolation_path):
            with open(isolation_path, 'rb') as f:
                self.isolation_model = pickle.load(f)
            print("✅ Isolation Forest chargé")
    
    def train_all(self, X, y):
        """Enchaîne toutes les étapes d'entraînement"""
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        X_resampled, y_resampled = self.apply_smote(X_train, y_train)
        
        self.train_logistic(X_resampled, y_resampled)
        self.train_isolation_forest(X_resampled)
        
        metrics_logistic = self.evaluate_model(
            self.logistic_model, X_test, y_test, 'Logistic Regression')
        metrics_isolation = self.evaluate_model(
            self.isolation_model, X_test, y_test, 'Isolation Forest')
        
        return metrics_logistic, metrics_isolation, X_test, y_test


if __name__ == "__main__":
    from data_processor import DataProcessor
    processor = DataProcessor()
    df, X, y = processor.process_all()
    
    models = FraudDetectionModels()
    metrics_lr, metrics_if, X_test, y_test = models.train_all(X, y)
    