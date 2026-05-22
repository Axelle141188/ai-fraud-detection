import pandas as pd
import numpy as np

class BehavioralAnalyzer:
    def __init__(self):
        self.risk_profiles = {
            'opportunist': {
                'name': 'Fraudeur Opportuniste',
                'description': 'Profite d\'une occasion unique, agit impulsivement',
                'indicators': ['low_amount', 'off_hours', 'single_transaction'],
                'color': '#ffaa00'
            },
            'methodical': {
                'name': 'Fraudeur Méthodique',
                'description': 'Planifie soigneusement, teste avant d\'agir à grande échelle',
                'indicators': ['multiple_transactions', 'increasing_amounts', 'regular_hours'],
                'color': '#ff4444'
            },
            'impulsive': {
                'name': 'Fraudeur Impulsif',
                'description': 'Agit sans planification, montants élevés, horaires atypiques',
                'indicators': ['high_amount', 'night_hours', 'single_transaction'],
                'color': '#ff0000'
            },
            'insider': {
                'name': 'Menace Interne',
                'description': 'Connaissance du système, transactions discrètes et régulières',
                'indicators': ['low_amount', 'business_hours', 'multiple_transactions'],
                'color': '#9b59b6'
            }
        }
    def analyze_timing(self, time_seconds):
        """Analyse psychologique de l'heure de la transaction"""
        hour = (time_seconds % 86400) // 3600
        
        if 2 <= hour <= 5:
            return {
                'period': 'Nuit profonde',
                'risk_level': 'ÉLEVÉ',
                'psychological_note': 'Horaire typique des attaques automatisées ou fraudeurs internationaux évitant la détection humaine',
                'hour': int(hour)
            }
        elif 6 <= hour <= 9:
            return {
                'period': 'Tôt le matin',
                'risk_level': 'MODÉRÉ',
                'psychological_note': 'Peut indiquer un fraudeur dans un fuseau horaire différent ou testant avant les heures ouvrées',
                'hour': int(hour)
            }
        elif 10 <= hour <= 17:
            return {
                'period': 'Heures ouvrées',
                'risk_level': 'FAIBLE',
                'psychological_note': 'Horaire normal — si fraude confirmée, peut indiquer une menace interne',
                'hour': int(hour)
            }
        else:
            return {
                'period': 'Soirée',
                'risk_level': 'MODÉRÉ',
                'psychological_note': 'Période de transition — fraudeurs exploitent la baisse de vigilance des équipes',
                'hour': int(hour)
            }
    def analyze_amount(self, amount, avg_fraud_amount=122.21):
        """Analyse psychologique du montant de la transaction"""
        if amount < 10:
            return {
                'category': 'Micro-transaction',
                'risk_level': 'MODÉRÉ',
                'psychological_note': 'Technique classique de test — le fraudeur vérifie si la carte est active avant une attaque plus importante',
                'ratio_to_avg': round(amount / avg_fraud_amount, 2)
            }
        elif amount < 100:
            return {
                'category': 'Transaction faible',
                'risk_level': 'FAIBLE',
                'psychological_note': 'Montant discret pour éviter les alertes automatiques des systèmes de détection',
                'ratio_to_avg': round(amount / avg_fraud_amount, 2)
            }
        elif amount < 500:
            return {
                'category': 'Transaction moyenne',
                'risk_level': 'MODÉRÉ',
                'psychological_note': 'Zone de confort du fraudeur méthodique — maximise le gain tout en restant sous les seuils d\'alerte',
                'ratio_to_avg': round(amount / avg_fraud_amount, 2)
            }
        else:
            return {
                'category': 'Transaction élevée',
                'risk_level': 'ÉLEVÉ',
                'psychological_note': 'Comportement impulsif ou fraudeur expérimenté très confiant — urgence de bloquer immédiatement',
                'ratio_to_avg': round(amount / avg_fraud_amount, 2)
            }
    def get_behavioral_profile(self, amount, time_seconds):
        """Détermine le profil psychologique du fraudeur"""
        timing = self.analyze_timing(time_seconds)
        amount_analysis = self.analyze_amount(amount)
        hour = timing['hour']
        
        is_night = hour >= 22 or hour <= 5
        is_business_hours = 9 <= hour <= 17
        is_high_amount = amount >= 500
        is_low_amount = amount < 50
        is_micro = amount < 10
        
        # Logique de profiling
        if is_micro:
            profile_key = 'methodical'
        elif is_high_amount and is_night:
            profile_key = 'impulsive'
        elif is_low_amount and is_business_hours:
            profile_key = 'insider'
        elif not is_night and not is_high_amount:
            profile_key = 'opportunist'
        else:
            profile_key = 'methodical'
        
        profile = self.risk_profiles[profile_key].copy()
        profile['timing_analysis'] = timing
        profile['amount_analysis'] = amount_analysis
        
        return profile
    def calculate_risk_score(self, amount, time_seconds, ml_confidence):
        """Calcule un score de risque hybride sur 100"""
        timing = self.analyze_timing(time_seconds)
        amount_analysis = self.analyze_amount(amount)
        
        # Score basé sur le timing
        timing_scores = {
            'ÉLEVÉ': 40,
            'MODÉRÉ': 20,
            'FAIBLE': 5
        }
        timing_score = timing_scores.get(timing['risk_level'], 10)
        
        # Score basé sur le montant
        amount_scores = {
            'ÉLEVÉ': 30,
            'MODÉRÉ': 15,
            'FAIBLE': 5
        }
        amount_score = amount_scores.get(amount_analysis['risk_level'], 10)
        
        # Score ML (0-30 points)
        ml_score = ml_confidence * 30
        
        # Score total sur 100
        total_score = min(100, timing_score + amount_score + ml_score)
        
        if total_score >= 75:
            risk_level = 'CRITIQUE'
            color = '#ff0000'
            action = 'Bloquer immédiatement et contacter le client'
        elif total_score >= 50:
            risk_level = 'ÉLEVÉ'
            color = '#ff4444'
            action = 'Vérification manuelle urgente requise'
        elif total_score >= 25:
            risk_level = 'MODÉRÉ'
            color = '#ffaa00'
            action = 'Surveiller les prochaines transactions'
        else:
            risk_level = 'FAIBLE'
            color = '#00ff88'
            action = 'Aucune action immédiate requise'
        
        return {
            'total_score': round(total_score, 1),
            'risk_level': risk_level,
            'color': color,
            'action': action,
            'breakdown': {
                'timing_score': timing_score,
                'amount_score': amount_score,
                'ml_score': round(ml_score, 1)
            }
        }
    def generate_behavioral_report(self, amount, time_seconds, ml_confidence):
        """Génère un rapport comportemental complet"""
        profile = self.get_behavioral_profile(amount, time_seconds)
        risk_score = self.calculate_risk_score(amount, time_seconds, ml_confidence)
        
        report = {
            'behavioral_profile': profile,
            'risk_score': risk_score,
            'summary': f"""
ANALYSE COMPORTEMENTALE
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Profil détecté    : {profile['name']}
Score de risque   : {risk_score['total_score']}/100
Niveau de risque  : {risk_score['risk_level']}
Action recommandée: {risk_score['action']}

CONTEXTE PSYCHOLOGIQUE
━━━━━━━━━━━━━━━━━━━━━━━━━━━
{profile['description']}

ANALYSE DU TIMING
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Période : {profile['timing_analysis']['period']}
Note    : {profile['timing_analysis']['psychological_note']}

ANALYSE DU MONTANT
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Catégorie : {profile['amount_analysis']['category']}
Note      : {profile['amount_analysis']['psychological_note']}
"""
        }
        return report


if __name__ == "__main__":
    analyzer = BehavioralAnalyzer()
    report = analyzer.generate_behavioral_report(
        amount=9.50,
        time_seconds=7200,
        ml_confidence=0.85
    )
    print(report['summary'])
    
    