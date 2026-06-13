# Implémentez un modèle de scoring crédit

> **Parcours Data Scientist — OpenClassrooms**  
> Auteur : Saliou TCHACOULA

---

## Contexte

Ce projet est réalisé dans le cadre d'une mission de Data Scientist au sein de la société financière fictive **"Prêt à dépenser"**, qui propose des crédits à la consommation à des personnes ayant peu ou pas d'historique de prêt.

L'objectif est double :

- Construire un **modèle de scoring** capable de calculer automatiquement la probabilité qu'un client rembourse son crédit, et de classer la demande en *accordé* ou *refusé*.
- Développer un **dashboard interactif** permettant d'interpréter les prédictions du modèle et d'améliorer la connaissance client des chargés de relation clientèle.
- Mettre en **production** le modèle via une API REST, et déployer le dashboard qui consomme cette API.

---

## Architecture du projet

```
.
├── data/
│   ├── X.csv                      # Features du jeu de test
│   ├── df_application_test.csv    # Données clients de test
│   └── df_train.csv               # Données d'entraînement (avec TARGET)
├── models/
│   ├── lgbm.joblib                # Modèle LightGBM entraîné
│   └── scaler.joblib              # Scaler (normalisation)
├── fast_API.py                    # API de prédiction (FastAPI)
├── app.py                         # Dashboard interactif (Dash / Plotly)
└── README.md
```

---

## Méthodologie

### 1. Préparation des données

- Téléchargement des données Home Credit (Kaggle) et exploitation d'un kernel de référence pour l'ingénierie de variables.
- Traitement des **valeurs manquantes** : suppression des colonnes/lignes dépassant un seuil défini.
- Correction des anomalies : transformation de `DAYS_BIRTH` et `DAYS_EMPLOYED` (valeurs négatives / aberrantes).
- Analyse des **corrélations** entre features et variable cible.
- Étude de **multicolinéarité** via le calcul des facteurs VIF (Variance Inflation Factor) : suppression des features fortement corrélées (VIF > 5).

### 2. Entraînement du modèle

Le problème présente un **déséquilibre de classes** marqué (beaucoup plus de prêts remboursés que de défauts). Les approches mises en œuvre :

- Rééchantillonnage avec **SMOTE** pour équilibrer les classes.
- Tests de plusieurs algorithmes : `RandomForestClassifier`, `LogisticRegression`, `XGBClassifier`, **`LGBMClassifier`** (modèle retenu).
- **Fonction de coût métier** : le coût d'un faux négatif (mauvais client prédit bon → perte en capital) est supposé dix fois supérieur au coût d'un faux positif (bon client refusé → manque à gagner).
- Suivi des expériences et des hyperparamètres via **MLFlow**.

### 3. Interprétabilité

**Locale (par client) :**  
Valeurs SHAP calculées via `TreeExplainer`. Un graphique *force plot* identifie la contribution de chaque variable à la prédiction d'un client donné (positive = hausse du risque, négative = baisse du risque).

**Globale (modèle complet) :**  
Feature importance du modèle LGBM. Les variables les plus influentes sont notamment `EXT_SOURCE_1`, `EXT_SOURCE_2`, `AMT_REQ_CREDIT_BUREAU_YEAR` et `REGION_POPULATION_RELATIVE`.

### 4. Analyse du Data Drift

Détection de dérives significatives dans la distribution des données sur **10 des 66 colonnes** du dataset (≈ 15 % des features), dont `ACTIVE_MONTHS_BALANCE_SIZE_MEAN`, `ACTIVE_MONTHS_BALANCE_MIN_MIN` et `PAYMENT_RATE`.

---

## API (FastAPI)

Le modèle est exposé via une API REST construite avec **FastAPI**.

### Endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| `GET` | `/` | Message de bienvenue |
| `GET` | `/check_client_exists?id={id}` | Vérifie l'existence d'un client |
| `POST` | `/predict` | Retourne la probabilité de défaut (entre 0 et 1) |

### Exemple de requête

```bash
curl -X POST "https://<votre-api>/predict" \
  -H "Content-Type: application/json" \
  -d '{"id": 0}'
```

### Lancement en local

```bash
pip install fastapi uvicorn pandas joblib scikit-learn lightgbm
uvicorn fast_API:app --host 0.0.0.0 --port 8000
```

---

## Dashboard interactif (Dash)

Le dashboard est développé avec **Dash / Plotly** et **Dash Bootstrap Components**.

### Fonctionnalités

- **Saisie d'un identifiant client** pour obtenir sa prédiction en temps réel.
- **Jauge de risque** (0 → 1) avec code couleur : rouge (risque élevé), jaune (zone intermédiaire), vert (faible risque).
- **Distribution univariée** : histogramme d'une feature choisie avec positionnement du client sélectionné.
- **Analyse bivariée** : scatter plot de deux features avec mise en évidence du client.
- **Feature importance globale** : graphique à barres horizontales.
- **Interprétabilité locale** : force plot SHAP interactif pour le client sélectionné.

### Lancement en local

```bash
pip install dash dash-bootstrap-components dash-daq plotly shap lightgbm joblib requests
python app.py
```

Le dashboard sera disponible à l'adresse `http://127.0.0.1:8050`.

---

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| Modèle ML | LightGBM |
| API | FastAPI + Uvicorn |
| Dashboard | Dash + Plotly + Dash Bootstrap Components |
| Interprétabilité | SHAP (TreeExplainer) |
| Suivi des expériences | MLFlow |
| Déploiement API | Azure Web Apps |
| Sérialisation | joblib |

---

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/<votre-username>/<votre-repo>.git
cd <votre-repo>

# Installer les dépendances
pip install -r requirements.txt
```

**Exemple de `requirements.txt` :**

```
fastapi
uvicorn
dash
dash-bootstrap-components
dash-daq
plotly
pandas
scikit-learn
lightgbm
shap
joblib
requests
```

---

## Limites et améliorations envisagées

- Améliorer la sélection des features (méthodes plus avancées).
- Réaliser des tests de normalité sur les distributions.
- Tester d'autres algorithmes de classification.
- Enrichir les tests unitaires et augmenter la couverture de code.

---

## Auteur

**Saliou TCHACOULA** — Étudiant Data Scientist, OpenClassrooms  
Projet soutenu en décembre 2023.
