# Librairies
from joblib import load
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd

df = pd.read_csv("/home/saliou/Bureau/Projets/Projet7/fichierdata.csv")
feat_importance = pd.read_csv("/home/saliou/Bureau/Projets/Projet7/feat_importance.csv")
df_importance = feat_importance.groupby(["feature"]).mean()["importance"].reset_index().sort_values("importance", ascending=False)
columns_to_drop = df_importance[df_importance["importance"]  < 72]["feature"].reset_index(drop=True)
df_clean = df.drop(columns=columns_to_drop).reset_index(drop=True)
missing_percentages = df_clean.isna().mean().sort_values(ascending=False)
columns_to_drop = missing_percentages[missing_percentages > 0.6].index
df_clean = df_clean.drop(columns=columns_to_drop)
pourcentage_nan_par_ligne = df_clean.isnull().mean(axis=1)
df_clean['Pourcentage_manquant'] = pourcentage_nan_par_ligne
df_clean = df_clean[df_clean['Pourcentage_manquant'] < 0.60].reset_index(drop=True)
df_clean.drop(columns=["index", "Pourcentage_manquant"], inplace=True)

# chargement du modèle
loaded_model = load("/home/saliou/Bureau/Projets/Projet7/lr.joblib")
# Création du nouvelle instance fastAPI
app = FastAPI()
# Définition d'un objet (une classe) pour réaliser des requêtes
class requet_body(BaseModel):
  df_clean.columns
# Définition du chemin du point de terminaison(API)
@app.post("/predict")
# Définition de la fonction de prédiction
def predict(data : requet_body):
  # Nouvelles données lesquelles on fait la prédiction
  new_data = X_test
  # prédiction
  class_idx = loaded_model.predict(new_data)[0]
  # Je retourne l' avis favorable ou non au client
  return {"class": df_clean[class_idx]}
