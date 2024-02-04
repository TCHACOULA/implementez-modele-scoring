import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import dash
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc
import requests

external_stylesheets = [dbc.themes.BOOTSTRAP]

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


# Initialisation de l'application Dash
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Mise en page du tableau de bord
app.layout = html.Div(children=[
    html.H1(children='Tableau de Bord avec Prédictions'),
    html.Hr(),
    dash_table.DataTable(data=X_test.head(10).to_dict('records'), page_size=10),
    html.Hr(),
    html.Div(
        [
            dbc.Row(
                dbc.Col(
                    dcc.Input(
                        id="input-client",
                        type="text",
                        placeholder="Entrez l'id d'un client"
                    ),
                    md=3

                )
                
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(id="prediction", children=[])
                    )
                ]
            )
        ]
    ),
    html.Hr(),
    html.Label('Sélectionnez une colonne :'),
    dcc.Dropdown(options=X_test.columns[1:], value='CODE_GENDER', id='feature-dropdown'),
    dcc.Graph(figure={}, id='feature-distribution'),
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(options=X_test.columns[1:], value='CODE_GENDER', id='feature-dropdown-bi1')),
        dbc.Col(
            dcc.Dropdown(options=X_test.columns[1:], value='CODE_GENDER', id='feature-dropdown-bi2'))  
    ]),
    dcc.Graph(figure={}, id='feature-distribution-bi'),
    #dcc.Dropdown(options=X_test.columns[1:], value='CODE_GENDER', id='feature-score_metier'),
    #dcc.Graph(figure={}, id='feature-score_metier'),

])

@callback(
    Output(component_id='prediction', component_property='children'),
    Input(component_id='input-client', component_property='value')
)
def update_prediction(client_id):
    response = requests.post(f"http://127.0.0.1:8000/predict?id={int(client_id)}")
    return "Résultat de la prédiction: {}".format(str(response.json()))

# Définition des callbacks pour les mises à jour dynamiques
@callback(
    Output(component_id='feature-distribution', component_property='figure'),
    Input(component_id='feature-dropdown', component_property='value')
)
def update_graph(col_chosen):
    fig = px.histogram(df_clean_train, x=col_chosen, color="TARGET", title=f'Distribution en fonction de {col_chosen}')
    return fig


@callback(
    Output(component_id='feature-distribution-bi', component_property='figure'),
    Input(component_id='feature-dropdown-bi1', component_property='value'),
    Input(component_id='feature-dropdown-bi2', component_property='value')
)
def update_graph_bi(feature_bi1, feature_bi2):
    fig = px.scatter(df_clean_train, x=feature_bi1, y=feature_bi2, color="TARGET", title=f'Analyse bivariée en fonction de {feature_bi1} et {feature_bi2}')
    return fig

#Définition des callbacks pour les mises à jour dynamiques
#@callback(
#    Output(component_id='feature-score_metier', component_property='figure'),
#    Input(component_id='score_metier', component_property='value')
#)
#def update_graph(col_chosen):
#    fig = px.histogram(df_clean_train, x=col_chosen, color="TARGET", title=f'Score en fonction de {col_chosen}')
#    return fig

# Lancement de l'application
if __name__ == '__main__':
    app.run_server(debug=True)

