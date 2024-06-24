# Librairies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from joblib import load
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

#df = pd.read_csv("/home/saliou/Bureau/Projets/Projet7/fichierdata.csv")
#feat_importance = pd.read_csv("/home/saliou/Bureau/Projets/Projet7/feat_importance.csv")
#df_importance = feat_importance.groupby(["feature"]).mean()["importance"].reset_index().sort_values("importance", ascending=False)
#columns_to_drop = df_importance[df_importance["importance"]  < 72]["feature"].reset_index(drop=True)
#df_clean = df.drop(columns=columns_to_drop).reset_index(drop=True)
#missing_percentages = df_clean.isna().mean().sort_values(ascending=False)
#columns_to_drop = missing_percentages[missing_percentages > 0.6].index
#df_clean = df_clean.drop(columns=columns_to_drop)
#pourcentage_nan_par_ligne = df_clean.isnull().mean(axis=1)
#df_clean['Pourcentage_manquant'] = pourcentage_nan_par_ligne
#df_clean = df_clean[df_clean['Pourcentage_manquant'] < 0.60].reset_index(drop=True)
#df_clean.drop(columns=["index", "Pourcentage_manquant"], inplace=True)
#df_application_train = df_clean[df_clean['TARGET'].notnull()]
#df_application_test = df_clean[df_clean['TARGET'].isnull()].reset_index(drop=True)
#X = df_application_train.drop('TARGET', axis=1)
#medians = X.median()  # Calculer la médiane de chaque colonne
#X.replace(np.inf, medians, inplace=True)
#X.replace(-np.inf, medians, inplace=True)
#X = X.fillna(medians)
#y = df_application_train['TARGET']
#df_application_test = df_application_test.drop('TARGET', axis=1)
#medians = df_application_test.median()  # Calculer la médiane de chaque colonne
#df_application_test.replace(np.inf, medians, inplace=True)
#df_application_test.replace(-np.inf, medians, inplace=True)
#df_application_test = df_application_test.fillna(medians)

X = pd.read_csv("~/oc-projects/implementez-modele-scoring/X.csv")
df_application_test = pd.read_csv("~/oc-projects/implementez-modele-scoring/df_application_test.csv")
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# chargement du modèle
#loaded_model = load("~/oc-projects/implementez-modele-scoring/lgbm.joblib")
loaded_model = load("/home/saliou/oc-projects/implementez-modele-scoring/lgbm.joblib")
# Création du nouvelle instance fastAPI
app = FastAPI()

# Définition de la fonction de prédiction
@app.post("/predict")
def predict(id: int):
    try:
        # Utilisez les données fournies dans la requête pour la prédiction
        client = df_application_test.iloc[id].values.reshape(1, -1)
        # Prédiction
        class_idx = loaded_model.predict(client)[0]

        # Retourne la classe prédite
        return class_idx
    except Exception as e:
        # En cas d'erreur, renvoie une réponse d'erreur
        return {"error": str(e)}

# Lancement de l'application
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)