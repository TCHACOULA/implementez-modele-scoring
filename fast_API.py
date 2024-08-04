# Librairies

import pandas as pd
from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel

class Client(BaseModel):
    id: int


X = pd.read_csv("./data/X.csv")
df_application_test = pd.read_csv("./data/df_application_test.csv")
client_ids = df_application_test.index.to_list()

# chargement du modèle
loaded_model = load("./models/lgbm.joblib")
loaded_scaler = load("./models/scaler.joblib")
# Création du nouvelle instance fastAPI
app = FastAPI()


@app.get("/")
async def read_main():
    return {"msg": "Bienvenue dans l'API du projet 7"}

@app.get("/check_client_exists")
def check_client_exists(id: int):
    if id in client_ids:
        return True
    else:
        return False



# Définition de la fonction de prédiction
@app.post("/predict")
def predict(c: Client):
    try:
        # Utilisez les données fournies dans la requête pour la prédiction
        client = df_application_test.iloc[c.id].values.reshape(1, -1)
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
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0")
