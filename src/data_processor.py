import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import streamlit as st

class DataProcessor:
    def __init__(self):
        self.data_dir = 'data'
        self.filepath = os.path.join(self.data_dir, 'creditcard.csv')
        self.scaler = StandardScaler()

    def download_from_kaggle(self):
        """Télécharge le dataset depuis Kaggle si non présent"""
        if os.path.exists(self.filepath):
            print("✅ Dataset déjà présent localement")
            return True

        try:
            import kaggle
            
            # Configurer les credentials depuis les secrets Streamlit
            os.environ['KAGGLE_USERNAME'] = st.secrets['KAGGLE_USERNAME']
            os.environ['KAGGLE_KEY'] = st.secrets['KAGGLE_KEY']

            print("⬇️ Téléchargement du dataset depuis Kaggle...")
            os.makedirs(self.data_dir, exist_ok=True)

            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                'mlg-ulb/creditcardfraud',
                path=self.data_dir,
                unzip=True
            )
            print("✅ Dataset téléchargé avec succès !")
            return True

        except Exception as e:
            print(f"❌ Erreur téléchargement Kaggle : {e}")
            return False

    def load_data(self):
        """Charge le dataset creditcard.csv"""
        # Télécharger si nécessaire
        if not os.path.exists(self.filepath):
            success = self.download_from_kaggle()
            if not success:
                print("❌ Impossible de charger le dataset")
                return pd.DataFrame()

        try:
            df = pd.read_csv(self.filepath)
            print(f"✅ Dataset chargé : {len(df)} transactions")
        except Exception:
            # Essayer avec séparateur point-virgule
            try:
                df = pd.read_csv(self.filepath, sep=';')
                df.columns = [col.replace('"', '').replace("'", '').strip() 
                              for col in df.columns]
                print(f"✅ Dataset chargé (sep=;) : {len(df)} transactions")
            except Exception as e:
                print(f"❌ Erreur lecture CSV : {e}")
                return pd.DataFrame()

        if 'Class' not in df.columns:
            print(f"❌ Colonne 'Class' introuvable")
            return pd.DataFrame()

        print(f"Fraudes : {df['Class'].sum()}")
        return df

    def clean_data(self, df):
        """Nettoie et vérifie les données"""
        df = df.drop_duplicates()
        df = df.dropna()

        expected_cols = ['Time', 'Amount', 'Class']
        for col in expected_cols:
            if col not in df.columns:
                print(f"❌ Colonne manquante : {col}")
                return pd.DataFrame()

        print(f"✅ Données nettoyées : {len(df)} transactions conservées")
        return df

    def prepare_features(self, df):
        """Normalise les données et sépare features et target"""
        df_ml = df.copy()
        df_ml['Amount_scaled'] = self.scaler.fit_transform(df_ml[['Amount']])
        df_ml['Time_scaled'] = self.scaler.fit_transform(df_ml[['Time']])

        # Supprimer les colonnes originales non normalisées
        df_ml = df_ml.drop(['Amount', 'Time'], axis=1)

        # Séparer features et target
        X = df_ml.drop('Class', axis=1)
        y = df_ml['Class']

        print(f"✅ Features préparées : {X.shape[1]} variables")
        return X, y

    def get_quick_stats(self, df):
        """Calcule les statistiques clés pour le dashboard
        IMPORTANT: df doit contenir la colonne Amount originale (non normalisée)
        """
        stats = {
            'total_transactions': len(df),
            'total_fraud': int(df['Class'].sum()),
            'fraud_rate': round(df['Class'].mean() * 100, 4),
            'avg_fraud_amount': round(df[df['Class']==1]['Amount'].mean() * 250.1196703376102 + 88.34961929394436, 2),
            'avg_normal_amount': round(df[df['Class']==0]['Amount'].mean() * 250.1196703376102 + 88.34961929394436, 2),
            'max_fraud_amount': round(df[df['Class']==1]['Amount'].max() * 250.1196703376102 + 88.34961929394436, 2)
        }
        return stats

    def process_all(self):
        """Enchaîne toutes les étapes en une seule commande
        Retourne df_original (avec Amount en euros), X (features ML), y (target)
        """
        df = self.load_data()
        if df.empty:
            return None, None, None
        df = self.clean_data(df)

        # df conserve Amount original en euros — utilisé pour les stats
        # X, y préparés séparément pour le ML
        X, y = self.prepare_features(df)

        return df, X, y


if __name__ == "__main__":
    processor = DataProcessor()
    df, X, y = processor.process_all()
    if df is not None:
        stats = processor.get_quick_stats(df)
        print(stats)

