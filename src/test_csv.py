import pandas as pd

# Lecture simple
df = pd.read_csv('data/creditcard.csv')

# Nettoyer les guillemets des noms de colonnes
df.columns = [c.strip('"') for c in df.columns]

print('Colonnes:', len(df.columns))
print('Dernieres colonnes:', df.columns[-3:].tolist())
print('Shape:', df.shape)

# Sauvegarder proprement
df.to_csv('data/creditcard_clean.csv', index=False)
print('Fichier propre sauvegarde!')