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

X = pd.read_csv("~/oc-projects/implementez-modele-scoring/X.csv")
df_application_test = pd.read_csv("~/oc-projects/implementez-modele-scoring/df_application_test.csv")


# chargement du modèle
loaded_model = load("/home/saliou/oc-projects/implementez-modele-scoring/lgbm.joblib")
loaded_scaler = load("/home/saliou/oc-projects/implementez-modele-scoring/scaler.joblib")
# Création du nouvelle instance fastAPI
app = FastAPI()

# Définition de la fonction de prédiction
@app.post("/predict")
def predict(id: int):
    try:
        # Utilisez les données fournies dans la requête pour la prédiction
        client = df_application_test.iloc[id].values.reshape(1, -1)
        client = loaded_scaler.transform(client)
        # Prédiction
        proba = loaded_model.predict_proba(client)[0][1]
        print(proba)

        # Retourne la classe prédite
        return proba
    except Exception as e:
        # En cas d'erreur, renvoie une réponse d'erreur
        return {"error": str(e)}

# Lancement de l'application
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)