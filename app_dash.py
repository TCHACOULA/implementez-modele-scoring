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
import dash_daq as daq
import requests
import shap
import base64
import plotly.graph_objects as go
import logging
from joblib import load

logging.basicConfig(level=logging.INFO)


external_stylesheets = [dbc.themes.BOOTSTRAP]

X = pd.read_csv("~/oc-projects/implementez-modele-scoring/X.csv")
df_application_test = pd.read_csv("~/oc-projects/implementez-modele-scoring/df_application_test.csv")
df_train = pd.read_csv("~/oc-projects/implementez-modele-scoring/df_train.csv")
    
# Initialisation de l'application Dash
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Mise en page du tableau de bord
app.layout = html.Div(children=[
    html.H1(children='Tableau de Bord avec Prédictions'),
    html.Hr(),
    dash_table.DataTable(data=df_application_test.head(10).to_dict('records'), page_size=10),
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
    html.Div([
        daq.Gauge(id='my-gauge-1',
                  label="Mon jauge", 
                  value=0, 
                  showCurrentValue=True,
                  units="MPH", 
                  max=1, 
                  min=0
                  ),
    ]),
    html.Hr(),
    html.Label('Sélectionnez une colonne :'),
    dcc.Dropdown(options=df_application_test.columns[1:], value='CODE_GENDER', id='feature-dropdown'),
    dcc.Graph(figure={}, id='feature-distribution'),
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(options=df_application_test.columns[1:], value='CODE_GENDER', id='feature-dropdown-bi1')),
        dbc.Col(
            dcc.Dropdown(options=df_application_test.columns[1:], value='CODE_GENDER', id='feature-dropdown-bi2'))  
    ]),
    dcc.Graph(figure={}, id='feature-distribution-bi'),
    html.Hr(),
    # Créer un composant graphique Dash à partir de la figure
    html.Div([
        dcc.Graph(
            id='global-importance',
            figure={}
        ),
    ]),
    html.Hr(),
    html.Div(id="local-importance"),
])

def predict(client_id):
    if not client_id:
        client_id = "0"
    response = requests.post(f"http://127.0.0.1:8000/predict?id={int(client_id)}")
    return str(response.json())

@callback(
    Output(component_id='prediction', component_property='children'),
    Input(component_id='input-client', component_property='value')
)
def update_prediction(client_id):
    pred = 0
    try:
        pred = predict(client_id)
    except Exception as e:
        logging.error("Une erreur est souvenue lors de la prédiction")

    return "Résultat de la prédiction: {}".format(pred)

@callback(
    Output(component_id='my-gauge-1', component_property='value'),
    Input(component_id='input-client', component_property='value')
)
def update_jauge(client_id):
    pred = 0
    try:
        pred = predict(client_id)
    except Exception as e:
        logging.error("Une erreur est souvenue lors de la prédiction")

    return float(pred)

# Définition des callbacks pour les mises à jour dynamiques
@callback(
    Output(component_id='feature-distribution', component_property='figure'),
    Input(component_id='feature-dropdown', component_property='value'),
    Input(component_id='input-client', component_property='value')
)
def update_graph(col_chosen, client_id):
    if not client_id:
        client_id = "0"
    fig = px.histogram(df_train, x=col_chosen, color="TARGET", title=f'Distribution en fonction de {col_chosen}')
    fig.add_vline(
        x=df_train.loc[int(client_id), col_chosen],
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
    if not client_id:
        client_id = "0"
    fig = px.scatter(df_train, x=feature_bi1, y=feature_bi2, color="TARGET", title=f'Analyse bivariée en fonction de {feature_bi1} et {feature_bi2}')
    fig.add_trace(px.scatter(
        x=[df_train.loc[int(client_id), feature_bi1]],
        y=[df_train.loc[int(client_id), feature_bi2]],
        color=[1],
        size=[100]).data[0])
    return fig


# Obtenir les valeurs SHAP et l'importance globale avec SHAP

# Créer un graphique à barres pour afficher l'importance globale
def create_global_importance_graph(feature_names):
    model = load("/home/saliou/oc-projects/implementez-modele-scoring/lgbm.joblib")
    df_feature_importance = (
        pd.DataFrame({
            'feature': model.feature_name_,
            'importance': model.feature_importances_,
        })
    .sort_values('importance', ascending=True)
    )
    features = [feature_names[int(col.split("_")[1])] for col in df_feature_importance.feature]
    fig = px.bar(
        y=features, 
        x=df_feature_importance.importance,
        orientation="h",
        title='Global Feature Importance'
    )
    return fig

# Créer un graphique à barres pour afficher l'importance locale
def create_local_importance_graph(client_id):
    model = load("/home/saliou/oc-projects/implementez-modele-scoring/lgbm.joblib")
    explainer_raw = shap.TreeExplainer(model)
    shap_values = explainer_raw(df_application_test)
    class_idx = 1
    expected_value = explainer_raw.expected_value[class_idx]
    shap_value = shap_values[:, :, class_idx].values[client_id]
    print(shap_value)
    force_plot = shap.force_plot(
        base_value=expected_value,
        shap_values=shap_value,
        features=df_application_test.iloc[client_id, :],
        link="logit",  # <-- here
        show=False
        )
    # Save the SHAP plot as a PNG image
    shap_image_path = 'shap_plot.html'
    shap.save_html(shap_image_path, force_plot)
    plt.close('all')

    iframe = html.Iframe(srcDoc=open(shap_image_path, 'r').read(), width='100%', height='600px')

    return iframe

# Définition des callbacks pour mettre à jour les graphiques d'importance globale et locale
@callback(
    Output(component_id='global-importance', component_property='figure'),
    Input(component_id='input-client', component_property='value')
)
def global_feature_importance(client_id):

    # Créer les graphiques d'importance globale et locale
    global_fig = create_global_importance_graph(df_application_test.columns.to_list())
    
    return global_fig

@callback(
    Output(component_id='local-importance', component_property='children'),
    Input(component_id='input-client', component_property='value')
)
def update_local_feature_importance(client_id):
    if not client_id:
        client_id = "0"

    # Créer les graphiques d'importance globale et locale
    local_fig = create_local_importance_graph(int(client_id))  # Afficher les valeurs pour le premier échantillon
    
    return local_fig

# Lancement de l'application
if __name__ == '__main__':
    app.run_server(debug=True)

