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
import shap

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
                        placeholder="Entrez l'id d'un client",
                        value="0"
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
    html.Hr(),

    # Créer un composant graphique Dash à partir de la figure
    html.Div([
        dcc.Graph(
            id='global-importance',
            figure={}
        ),
        dcc.Graph(
            id='local-importance',
            figure={}  # Afficher les valeurs pour le premier échantillon
        ),

    ]),
   
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
    Input(component_id='feature-dropdown', component_property='value'),
    Input(component_id='input-client', component_property='value')
)
def update_graph(col_chosen, client_id):
    fig = px.histogram(df_clean_train, x=col_chosen, color="TARGET", title=f'Distribution en fonction de {col_chosen}')
    fig.add_vline(
        x=df_clean_train.loc[int(client_id), col_chosen],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Client id: {client_id}",
        annotation_position="top right"
    )
    return fig


@callback(
    Output(component_id='feature-distribution-bi', component_property='figure'),
    Input(component_id='feature-dropdown-bi1', component_property='value'),
    Input(component_id='feature-dropdown-bi2', component_property='value'),
    Input(component_id='input-client', component_property='value')
)
def update_graph_bi(feature_bi1, feature_bi2, client_id):
    fig = px.scatter(df_clean_train, x=feature_bi1, y=feature_bi2, color="TARGET", title=f'Analyse bivariée en fonction de {feature_bi1} et {feature_bi2}')
    fig.add_trace(px.scatter(
        x=[df_clean_train.loc[int(client_id), feature_bi1]],
        y=[df_clean_train.loc[int(client_id), feature_bi2]],
        color=[1],
        size=[100]).data[0])
    return fig

# Fonction pour entraîner un modèle et obtenir les importances des fonctionnalités
def train_model_and_get_feature_importance(data, client_id):
    # Filtrer les données en fonction du client spécifié
    #filtered_data = data.loc[client_id]
    
    # Séparation des caractéristiques et de la cible
    X = data.drop('TARGET', axis=1)
    medians = X.median()  
    X.replace(np.inf, medians, inplace=True)
    X.replace(-np.inf, medians, inplace=True)
    X = X.fillna(medians)
    y = data['TARGET']
    
    # Création du modèle
    model = RandomForestClassifier()
    
    # Entraînement du modèle
    model.fit(X, y)
    
    # Initialiser l'explorateur SHAP avec le modèle entraîné
    explainer = shap.TreeExplainer(model)
    
    # Calculer les valeurs SHAP pour l'ensemble des données
    shap_values = explainer.shap_values(X)
    
    # Obtention de l'importance globale
    
    global_shap_df = (
        pd.DataFrame(np.abs(shap_values.values)
        .mean(axis=0), index=X.columns, columns=['SHAP Value'])
        .sort_values(by='SHAP Value', ascending=False)
    )
    print(global_shap_df)
    local_shap_values = (
        pd.DataFrame(shap_values.values[client_id], index=X.columns, columns=['SHAP Value'])
        .sort_values(by='SHAP Value', ascending=False)
    )
    print(local_shap_values)
    return local_shap_values, global_shap_df


# Obtenir les valeurs SHAP et l'importance globale avec SHAP
#shap_values, global_importance = train_model_and_get_feature_importance(df_clean_train, client_id=client_id)

# Créer un graphique à barres pour afficher l'importance globale
def create_global_importance_graph(global_importance):
    fig = px.bar(
        global_importance,
        y=global_importance.index,
        x="Shap value",
        orientation="h",
        #text=[global_importance],
        #labels={'x': 'Importance', 'y': 'Value'},
        title='Global Feature Importance'
    )
    return fig

# Créer un graphique à barres pour afficher l'importance locale
def create_local_importance_graph(shap_values, client_id):
    fig = px.bar(
        shap_values,
        y=shap_values.index,
        x="Shap value",
        orientation="h",
        #text=['Feature ' + str(i) for i in range(len(shap_values[index]))],
        #labels={'x': 'Feature Index', 'y': 'Importance'},
        title='Local Feature Importance (Sample ' + str(client_id) + ')'
    )
    return fig

# Définition des callbacks pour mettre à jour les graphiques d'importance globale et locale
@callback(
    Output(component_id='global-importance', component_property='figure'),
    Output(component_id='local-importance', component_property='figure'),
    Input(component_id='input-client', component_property='value')
)
def update_feature_importance(client_id):
    # Entraîner le modèle et obtenir les importances des fonctionnalités
    shap_values, global_importance = train_model_and_get_feature_importance(df_clean_train, int(client_id))
    print(shap_values)
    print(global_importance)
    # Créer les graphiques d'importance globale et locale
    global_fig = create_global_importance_graph(global_importance)
    local_fig = create_local_importance_graph(shap_values, int(client_id))  # Afficher les valeurs pour le premier échantillon
    
    return global_fig, local_fig


# Lancement de l'application
if __name__ == '__main__':
    app.run_server(debug=True)

