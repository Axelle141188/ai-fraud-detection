# 🔴 AI Fraud Detection System

A machine learning system that detects fraudulent banking transactions
by combining two complementary ML models with behavioral psychology analysis.

> Built for the banking and financial sector, where fraud detection combines
> technical pattern recognition with human behavioral understanding.

##  Features

- Dual ML detection : Logistic Regression (supervised) +
  Isolation Forest (unsupervised)
- SMOTE implementation to handle extreme class imbalance (0.17% fraud rate)
- Behavioral Risk Profiling : psychological analysis of each suspicious transaction
- Real-time risk scoring system (0-100) combining ML confidence + behavioral signals
- Four fraudster psychological profiles : Opportunist, Methodical, Impulsive, Insider
- Interactive transaction analyzer with visual risk gauge
- Modular architecture (processor / models / behavioral / dashboard)

##  Tech Stack

| Tool | Usage |
|------|-------|
| Python 3.x | Core language |
| Scikit-learn | ML models (Logistic Regression, Isolation Forest) |
| Imbalanced-learn | SMOTE for class imbalance |
| Streamlit | Interactive dashboard |
| Plotly | Data visualizations & risk gauge |
| Pandas / NumPy | Data processing |

##  Installation

```bash
# 1. Clone le repository
git clone https://github.com/TON_USERNAME/ai-fraud-detection
cd ai-fraud-detection

# 2. Installe les dépendances
pip install -r requirements.txt

# 3. Télécharge le dataset
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place creditcard.csv dans le dossier data/

# 4. Lance le dashboard
streamlit run src/dashboard.py
```

Ouvre ton navigateur sur **http://localhost:8501**

>  Premier lancement : les modèles s'entraînent automatiquement
> (environ 2-3 minutes). Les lancements suivants sont instantanés.

##  Project Structure

```
ai-fraud-detection/
├── data/
│   ├── creditcard.csv          # Kaggle dataset (284,807 transactions)
│   ├── logistic_model.pkl      # Trained model (auto-generated)
│   └── isolation_model.pkl     # Trained model (auto-generated)
├── src/
│   ├── data_processor.py       # Data loading, cleaning & feature engineering
│   ├── models.py               # ML models training & evaluation
│   ├── behavioral.py           # Psychological profiling & risk scoring
│   └── dashboard.py            # Streamlit visual interface
├── requirements.txt
└── README.md
```

##  Model Performance

| Metric | Logistic Regression | Isolation Forest |
|--------|-------------------|-----------------|
| Precision | 91.3% | 25.8% |
| Recall | 91.3% | 25.8% |
| F1-Score | 91.3% | 25.8% |
| ROC-AUC | 95.6% | 62.8% |

**Why two models ?**
- Logistic Regression excels at detecting known fraud patterns → recommended for production
- Isolation Forest detects unknown anomalies without labeled data → useful for emerging threats

**The class imbalance challenge :**
Only 0.17% of transactions are fraudulent (492 out of 284,807).
SMOTE was applied to balance the training data and prevent
the model from ignoring fraud cases.

##  Behavioral Analysis — The Unique Differentiator

Beyond ML metrics, this system profiles the human behavior behind each
suspicious transaction using psychological analysis.

### Four Fraudster Profiles

| Profile | Behavior | Key Indicators |
|---------|----------|----------------|
| 🟡 Opportunist | Acts impulsively on a single opportunity | Low amount, off-hours |
| 🔴 Methodical | Plans carefully, tests before scaling | Micro-transactions, increasing amounts |
| 🔴 Impulsive | No planning, high amounts, night hours | High amount, 2h-5h AM |
| 🟣 Insider | System knowledge, discrete & regular | Low amount, business hours |

### Risk Scoring System (0-100)
- **Timing analysis** → up to 40 points
- **Amount analysis** → up to 30 points
- **ML confidence** → up to 30 points

##  Dataset

- **Source :** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size :** 284,807 transactions (September 2013, European cardholders)
- **Features :** 30 variables (V1-V28 PCA-transformed + Time + Amount)
- **Fraud rate :** 0.17% (492 fraudulent transactions)

*Data anonymized via PCA transformation for confidentiality —
consistent with real banking data handling requirements.*

---

##  Author

**Axelle** — Aspiring Behavioral Fraud Detection Specialist

- 🎓 Background in Psychology | Google Data Analytics Certificate
- 🔐 CompTIA Security+ in preparation | CySA+ planned
- 🌍 French (native) | Portuguese (native) | English (working knowledge)
- 🎯 Targeting fraud detection roles in the banking and financial sector

*"Fraud isn't just algorithmic — it's human behavior with
detectable psychological patterns. This system bridges the gap
between technical detection and behavioral understanding."*
