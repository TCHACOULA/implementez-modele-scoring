import pandas as pd
import numpy as np
from sklearn import linear_model
import pickle
import flask
from flask import request
from sklearn.model_selection import train_test_split

# Création du df_clean
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

# Split des données
df_application_train = df_clean[df_clean['TARGET'].notnull()]
df_application_test = df_clean[df_clean['TARGET'].isnull()]
X = df_application_train.drop('TARGET', axis=1)
medians = X.median()  # Calculer la médiane de chaque colonne
X.replace(np.inf, medians, inplace=True)
X.replace(-np.inf, medians, inplace=True)
X = X.fillna(medians)
y = df_application_train['TARGET']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
df_clean_train = pd.concat([X, y], axis=1)
data = df_clean_train.head(10)

#loading and separating our wine dataset into labels and features
#application_train = pd.read_csv('/home/saliou/Bureau/Projets/Projet7/application_train.csv', delimiter=",")
label = data['TARGET']
features = data.drop('TARGET', axis=1)
#application_test = pd.read_csv('/home/saliou/Bureau/Projets/Projet7/application_test.csv')

#defining our linear regression estimator and training it with our wine data
regr = linear_model.LinearRegression()
print(features)
regr.fit(features, label)
#using our trained model to predict a fake wine
#each number represents a feature like pH, acidity, etc.
print(regr.predict(X_test.head(10)).tolist())
#creating and training a model
regr = linear_model.LinearRegression()
regr.fit(features, label)
#serializing our model to a file called model.pkl
pickle.dump(regr, open("/home/saliou/Bureau/Projets/Projet7/model.pkl","wb"))

#loading a model from a file called model.pkl
#model = pickle.load(open("model.pkl","r"))
model = pickle.load(open("/home/saliou/Bureau/Projets/Projet7/model.pkl", "rb"))

app = flask.Flask(__name__)
#getting our trained model from a file we created earlier
#model = pickle.load(open("model.pkl","r"))
model = pickle.load(open("/home/saliou/Bureau/Projets/Projet7/model.pkl", "rb"))

#defining a route for only post requests
@app.route('/predict', methods=['POST'])
def predict():
    #getting an array of features from the post request's body
    query_parameters = request.args
    feature_array = np.fromstring(query_parameters['feature_array'],dtype=float,sep=",")
    #creating a response object
    #storing the model's prediction in the object
    response = {}
    response['predictions'] = model.predict([feature_array]).tolist()
    #returning the response object as json
    return flask.jsonify(response)
if __name__ == "__main__":
    app.run(debug=True)