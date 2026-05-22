import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

class DataProcessor:
    def __init__(self):
        self.data_dir = 'data'
        self.filepath = os.path.join(self.data_dir, 'creditcard.csv')
        self.scaler = StandardScaler()
    def load_data(self):
        """Charge le dataset creditcard.csv"""
        if not os.path.exists(self.filepath):
            print("❌ Fichier creditcard.csv introuvable dans data/")
            return pd.DataFrame()
        
        df = pd.read_csv(self.filepath, sep=';')
        
        # Supprimer tous les guillemets des noms de colonnes
        df.columns = [col.replace('"', '').replace("'", '').strip() 
                      for col in df.columns]
        
        print(f"✅ Dataset chargé : {len(df)} transactions")
        
        if 'Class' not in df.columns:
            print(f"❌ Colonne 'Class' introuvable")
            return pd.DataFrame()
            
        print(f" Fraudes : {df['Class'].sum()}")
        return df
    def clean_data(self, df):
        """Nettoie et vérifie les données"""
        # Supprimer les doublons
        df = df.drop_duplicates()
        
        # Supprimer les valeurs manquantes
        df = df.dropna()
        
        # Vérifier les colonnes attendues
        expected_cols = ['Time', 'Amount', 'Class']
        for col in expected_cols:
            if col not in df.columns:
                print(f"❌ Colonne manquante : {col}")
                return pd.DataFrame()
        
        print(f"✅ Données nettoyées : {len(df)} transactions conservées")
        return df
    def prepare_features(self, df):
        """Normalise les données et sépare features et target"""
        df = df.copy()
        df['Amount_scaled'] = self.scaler.fit_transform(df[['Amount']])
        df['Time_scaled'] = self.scaler.fit_transform(df[['Time']])
        
        # Supprimer les colonnes originales non normalisées
        df = df.drop(['Amount', 'Time'], axis=1)
        
        # Séparer features et target
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        print(f"✅ Features préparées : {X.shape[1]} variables")
        return X, y
    def get_quick_stats(self, df):
        """Calcule les statistiques clés pour le dashboard"""
        stats = {
            'total_transactions': len(df),
            'total_fraud': int(df['Class'].sum()),
            'fraud_rate': round(df['Class'].mean() * 100, 4),
            'avg_fraud_amount': round(df[df['Class']==1]['Amount'].mean(), 2),
            'avg_normal_amount': round(df[df['Class']==0]['Amount'].mean(), 2),
            'max_fraud_amount': round(df[df['Class']==1]['Amount'].max(), 2)
        }
        return stats
    
    def process_all(self):
        """Enchaîne toutes les étapes en une seule commande"""
        df = self.load_data()
        if df.empty:
            return None, None, None
        df = self.clean_data(df)
        X, y = self.prepare_features(df)
        return df, X, y


if __name__ == "__main__":
    processor = DataProcessor()
    df, X, y = processor.process_all()
    stats = processor.get_quick_stats(df)
    print(stats)
    

