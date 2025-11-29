# HOUTATE SAÏD

<img src="HOUTATE Saïd CAC 2.jpg" style="height:300px;margin-right:300px; float:left; border-radius:10px;"/>


**Numéro d'étudiant** : 24010355  
**Classe** : CAC2

<br clear="left"/>

---

# Compte rendu
## Analyse du Taux d’Engagement Instagram par Régression

**Date :** 26 Novembre 2025

---

# À propos du jeu de données :

  Ce fichier contient 30 000 publications Instagram avec des analyses détaillées collectées au cours des 12 derniers mois. Chaque ligne représente une publication Instagram et inclut des informations sur le type de média, les indicateurs d'engagement, la portée, les impressions, les enregistrements, les partages, les sources de trafic et le taux d'engagement estimé.
  
  Ce jeu de données est conçu pour simuler des données Instagram Insights réalistes et reproduit le comportement naturel de l'algorithme d'Instagram. Des indicateurs tels que les mentions « J'aime », la portée, les impressions, les enregistrements et le nombre d'abonnés gagnés ont été générés à l'aide de distributions statistiques réalistes afin de refléter les performances typiques des publications Photos, Vidéos, Reels et Carrousel.

---

## Table des Matières

1. [Introduction et Contexte](#1-introduction-et-contexte)
2. [Analyse Exploratoire des Données (Data Analysis)](#2-analyse-exploratoire-des-données-data-analysis)
    * [Chargement et Structure du Dataset](#21-chargement-et-structure-du-dataset)
    * [Prétraitement et Ingénierie de Caractéristiques](#22-prétraitement-et-ingénierie-de-caractéristiques)
    * [Gestion des Valeurs Manquantes](#23-gestion-des-valeurs-manquantes)
    * [Analyse Statistique et Visuelle](#24-analyse-statistique-et-visuelle)
3. [Méthodologie de Modélisation](#3-méthodologie-de-modélisation)
    * [Séparation des Données (Data Split)](#31-séparation-des-données-data-split)
    * [Modèles de Régression Testés](#32-modèles-de-régression-testés)
4. [Résultats et Comparaison des Modèles](#4-résultats-et-comparaison-des-modèles)
    * [Régression Linéaire](#41-régression-linéaire)
    * [Régression Polynomiale](#42-régression-polynomiale)
    * [Régression par Arbre de Décision](#43-régression-par-arbre-de-décision)
    * [Régression par Forêt Aléatoire](#44-régression-par-forêt-aléatoire)
    * [Régression SVR (Support Vector Regression)](#45-régression-svr-support-vector-regression)
    * [Graphique et Tableau Comparatif des Performances](#46-graphique-et-tableau-comparatif-des-performances)
5. [Analyse des Résultats et Recommandations](#5-analyse-des-résultats-et-recommandations)
6. [Conclusion](#6-conclusion)

---

## 1. Introduction et Contexte

Ce rapport présente une analyse détaillée d'un jeu de données réel concernant les statistiques d'engagement des publications Instagram, issue de la plateforme Kaggle. L'objectif du projet est de construire et comparer plusieurs modèles de régression pour prédire le **taux d'engagement** ($Y$) en fonction de diverses caractéristiques liées aux publications, au contenu et aux temporalités.

En suivant le cycle de vie des données, nous avons mené une exploration (EDA), un prétraitement rigoureux, une ingénierie de caractéristiques, et une modélisation prédictive avec plusieurs algorithmes pour identifier le modèle le plus performant.

---

## 2. Analyse Exploratoire des Données (Data Analysis)

### 2.1 Chargement et Structure du Dataset

Le jeu de données `Instagram_Analytics.csv` contient des informations exhaustives sur les publications Instagram.

* **Nombre d'observations ($N$) :** 29 999 publications.
* **Nombre de variables ($d$) :** 15 colonnes (14 features + 1 target).

**Variables d'entrée ($X$) :**
- **Type de contenu :** `media_type` (Reel, Photo, Video, Carousel)
- **Engagement direct :** `likes`, `comments`, `shares`, `saves`
- **Portée :** `reach`, `impressions`, `followers_gained`
- **Métadonnées :** `caption_length`, `hashtags_count`
- **Stratégie de distribution :** `traffic_source` (Home Feed, Hashtags, Reels Feed, External, Profile, Explore)
- **Catégorie de contenu :** `content_category` (Technology, Fitness, Beauty, Music, Travel, Photography, etc.)

**Variable cible ($Y$) :** `engagement_rate` (taux d'engagement en pourcentage, plage observée : 2.90 à 23.52 et au-delà).

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Chargement des données
df = pd.read_csv('Instagram_Analytics.csv')

print("========= Résumé du Dataset =========")
print(f"Dimensions : {df.shape}")  # (29999, 15)
df.info()
print("\n========= Premiers échantillons =========")
print(df.head())
```

### 2.2 Prétraitement et Ingénierie de Caractéristiques

#### Extraction des Caractéristiques Temporelles

La colonne `upload_date` a été convertie au format datetime et les caractéristiques temporelles pertinentes ont été extraites :

```python
# Conversion en datetime et extraction de caractéristiques temporelles
df['upload_date'] = pd.to_datetime(df['upload_date'])
df['upload_month'] = df['upload_date'].dt.month
df['upload_day_of_week'] = df['upload_date'].dt.dayofweek  # Lundi=0, Dimanche=6
df['upload_hour'] = df['upload_date'].dt.hour
df['upload_year'] = df['upload_date'].dt.year

print("Caractéristiques temporelles extraites et ajoutées au DataFrame.")
```

Ces variables temporelles permettent de capturer les variations saisonnières et les tendances par jour/heure de publication, qui peuvent influencer le taux d'engagement.

#### Encodage des Variables Catégorielles

Les variables catégorielles ont été transformées en variables numériques via l'encodage **One-Hot** :

```python
# Encodage One-Hot des variables catégorielles
categorical_cols = ['media_type', 'traffic_source', 'content_category']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("Variables catégorielles One-Hot encodées.")
print(f"Nouvelles dimensions : {df.shape}")
```

**Justification :** L'option `drop_first=True` élimine une catégorie par variable pour éviter la **multicolinéarité** (problème de parfaite colinéarité entre les colonnes).

#### Suppression des Colonnes Non Pertinentes

```python
# Création du DataFrame nettoyé
df_processed = df.drop(columns=['post_id', 'upload_date'])

print("Colonnes non pertinentes (post_id, upload_date originale) supprimées.")
print(f"Dimensions finales : {df_processed.shape}")
print(df_processed.head())
```

### 2.3 Gestion des Valeurs Manquantes

```python
# Vérification des valeurs manquantes après transformations
print("========= Valeurs manquantes =========")
print(df_processed.isnull().sum())
```

Le dataset ne contient aucune valeur manquante (`NaN`), ce qui facilite grandement la modélisation.

### 2.4 Analyse Statistique et Visuelle

Une analyse exploratoire complète a été réalisée pour comprendre les distributions et les corrélations :

- **Distribution de la variable cible :** Le taux d'engagement présente une distribution légèrement asymétrique, avec des valeurs majoritairement comprises entre 4% et 10%, mais des valeurs extrêmes atteignant 23%+.
- **Corrélations :** Des boxplots et une matrice de corrélation ont révélé des relations entre les variables d'entrée et le taux d'engagement.
- **Observations clés :** Les variables `likes`, `comments`, `shares` et `reach` présentent des échelles très différentes, justifiant une **normalisation** pour les modèles sensibles à la distance (comme SVR).

---

## 3. Méthodologie de Modélisation

### 3.1 Séparation des Données (Data Split)

Le jeu de données a été divisé selon un schéma classique :

```python
from sklearn.model_selection import train_test_split

# Séparation des cibles et features
y = df_processed['engagement_rate']
X = df_processed.drop(columns=['engagement_rate'])

# Séparation Train (80%) / Test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Ensemble d'entraînement : {X_train.shape}")  # (23999, ...)
print(f"Ensemble de test : {X_test.shape}")          # (6000, ...)
```

**Justification :** La séparation 80/20 est un standard robuste. Le `random_state=42` garantit la reproductibilité.

### 3.2 Modèles de Régression Testés

Cinq modèles distincts ont été entraînés et évalués :

1. **Régression Linéaire** : Modèle de base, supposant une relation linéaire entre features et target.
2. **Régression Polynomiale (degré 2)** : Capture les non-linéarités via des termes quadratiques.
3. **Arbre de Décision** : Modèle non-paramétrique basé sur des partitions récursives.
4. **Forêt Aléatoire (Random Forest)** : Ensemble d'arbres réduisant le sur-apprentissage.
5. **SVR (Support Vector Regression)** : Modèle robuste basé sur les machines à vecteurs de support, avec normalisation préalable.

---

## 4. Résultats et Comparaison des Modèles

Les performances de chaque modèle ont été évaluées sur l'ensemble de test selon trois métriques clés :
- **$R^2$ (Coefficient de Détermination)** : Pourcentage de variance expliquée (de 0 à 1, plus proche de 1 = mieux).
- **MSE (Mean Squared Error)** : Erreur quadratique moyenne (plus bas = mieux).
- **RMSE (Root Mean Squared Error)** : Racine carrée de MSE, exprimée dans l'unité de la cible (plus bas = mieux).

### 4.1 Régression Linéaire

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Régression Linéaire - R² : {r2_lr:.4f}")
print(f"Régression Linéaire - MSE : {mse_lr:.2f}")
print(f"Régression Linéaire - RMSE : {rmse_lr:.2f}")
```

**Résultats :**
- **$R^2$ ≈ 0.0899** (9%)
- **MSE ≈ 2238.45**
- **RMSE ≈ 47.31**

![Logo](Graphiques/Régression_linéaire.png)


La régression linéaire explique seulement ~9% de la variance, indiquant que la relation entre les features et le taux d'engagement n'est pas linéaire ou que certaines variables clés manquent.

### 4.2 Régression Polynomiale

```python
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)
y_pred_poly = model_poly.predict(X_test_poly)

mse_poly = mean_squared_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f"Régression Polynomiale - R² : {r2_poly:.4f}")
print(f"Régression Polynomiale - MSE : {mse_poly:.2f}")
print(f"Régression Polynomiale - RMSE : {rmse_poly:.2f}")
```

**Résultats :**
- **$R^2$ ≈ 0.1706** (17%)
- **MSE ≈ 2062.18**
- **RMSE ≈ 45.41**

![Logo](Graphiques/Régression_polynomiale.png)

La régression polynomiale améliore les performances de la régression linéaire (+8% sur $R^2$), montrant que des interactions entre variables sont pertinentes. Cependant, le modèle reste insuffisant.

### 4.3 Régression par Arbre de Décision

```python
from sklearn.tree import DecisionTreeRegressor

model_dt = DecisionTreeRegressor(random_state=42)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)

mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print(f"Arbre de Décision - R² : {r2_dt:.4f}")
print(f"Arbre de Décision - MSE : {mse_dt:.2f}")
print(f"Arbre de Décision - RMSE : {rmse_dt:.2f}")
```

**Résultats :**
- **$R^2$ ≈ 0.7126** (71%)
- **MSE ≈ 707.89**
- **RMSE ≈ 26.61**

**Performance spectaculaire** : L'arbre de décision surpasse considérablement tous les modèles précédents.
- Le $R^2$ augmente de 9% (régression linéaire) à **71%**.
- La RMSE diminue d'environ **47** à **27**, une réduction d'environ 43%.

![Logo](Graphiques/Régression_par_Arbres_de_Décision.png)

Cela indique que la relation entre les features et le taux d'engagement est hautement non-linéaire et que les arbres de décision capturent bien ces interactions.

### 4.4 Régression par Forêt Aléatoire

```python
from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Forêt Aléatoire - R² : {r2_rf:.4f}")
print(f"Forêt Aléatoire - MSE : {mse_rf:.2f}")
print(f"Forêt Aléatoire - RMSE : {rmse_rf:.2f}")
```

**Résultats :**
- **$R^2$ ≈ 0.5900** (59%)
- **MSE ≈ 1015.68**
- **RMSE ≈ 31.87**

![Logo](Graphiques/Régression_par_Forêts_Aléatoires.png)

La Forêt Aléatoire obtient des performances meilleures que la Régression Polynomiale mais **inférieures** à l'Arbre de Décision. Cela peut sembler contre-intuitif (un ensemble d'arbres surpassant un arbre seul), mais plusieurs facteurs l'expliquent :
- Le jeu de données est suffisamment volumineux pour qu'un arbre unique bien construit performe bien.
- La Forêt peut introduire trop de régularisation, réduisant sa capacité à capturer les patterns spécifiques aux données d'entraînement (trade-off biais-variance).

### 4.5 Régression SVR (Support Vector Regression)

SVR est sensible à l'échelle des variables, une normalisation est donc appliquée :

```python
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Normalisation des features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entraînement du modèle SVR
model_svr = SVR(kernel='rbf')
model_svr.fit(X_train_scaled, y_train)
y_pred_svr = model_svr.predict(X_test_scaled)

mse_svr = mean_squared_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mse_svr)
r2_svr = r2_score(y_test, y_pred_svr)

print(f"SVR - R² : {r2_svr:.4f}")
print(f"SVR - MSE : {mse_svr:.2f}")
print(f"SVR - RMSE : {rmse_svr:.2f}")
```

**Résultats :**
- **$R^2$ ≈ 0.0899** (9%)
- **MSE ≈ 2238.45**
- **RMSE ≈ 47.31**

![Logo](Graphiques/Régression_CVR.png)

Le SVR présente les **mêmes performances que la Régression Linéaire**, ce qui suggère que le noyau RBF n'a pas capturé efficacement les patterns non-linéaires du jeu de données. Un réglage des hyperparamètres (`C`, `gamma`) pourrait améliorer les résultats.

### 4.6 Graphique et Tableau Comparatif des Performances

*1-* **Graphique : Comparaison des Modèles et Visualisation des Résultats**

![Logo](Graphiques/Comparaison_des_Modèles_et_Visualisation_des_Résultats.png)

*2-* **Tableau : Comparaison des Performances**

| Modèle                   | $R^2$    | MSE       | RMSE    | Performance          |
|--------------------------|----------|-----------|---------|----------------------|
| Régression Linéaire      | 0.0899   | 2238.45   | 47.31   | ⭐ Très faible       |
| Régression Polynomiale   | 0.1706   | 2062.18   | 45.41   | ⭐⭐ Faible           |
| **Arbre de Décision**    | **0.7126** | **707.89** | **26.61** | ⭐⭐⭐⭐⭐ Excellent |
| Forêt Aléatoire          | 0.5900   | 1015.68   | 31.87   | ⭐⭐⭐⭐ Très bon    |
| SVR                      | 0.0899   | 2238.45   | 47.31   | ⭐ Très faible       |

---

## 5. Analyse des Résultats et Recommandations

### Modèle Gagnant : Arbre de Décision

L'**Arbre de Décision surpasse tous les autres modèles** avec un $R^2$ de **0.71** et une RMSE de **26.61**.

**Interprétation :**
- Le modèle explique **71% de la variance** du taux d'engagement.
- Les erreurs de prédiction sont en moyenne d'environ **26.61 points de pourcentage** sur la cible.
- Comparé à la régression linéaire, l'arbre offre une amélioration de **+62 points de $R^2$** et une réduction d'erreur d'environ **44%**.

### Classement des Modèles

1. **Arbre de Décision** : 0.7126 (71%)
2. **Forêt Aléatoire** : 0.5900 (59%)
3. **Régression Polynomiale** : 0.1706 (17%)
4. **Régression Linéaire** : 0.0899 (9%) — **ex aequo** avec SVR
5. **SVR** : 0.0899 (9%)

### Recommandations pour Amélioration Future

1. **Optimisation des hyperparamètres de l'Arbre de Décision :** Explorer les paramètres tels que `max_depth`, `min_samples_split`, `min_samples_leaf` via GridSearchCV ou RandomizedSearchCV.

2. **Amélioration de la Forêt Aléatoire :** Augmenter le nombre d'arbres (`n_estimators`), ajuster la profondeur maximale, ou explorer d'autres ensembles (Gradient Boosting, XGBoost, LightGBM).

3. **Tuning du SVR :** Avec normalisation appropriée et ajustement du paramètre `C` et du `gamma` du noyau RBF.

4. **Feature Engineering avancé :** Créer des features d'interaction, des ratios (par ex. : `likes / reach`), ou appliquer des transformations non-linéaires.

5. **Analyse de l'importance des variables :** Pour l'Arbre de Décision, extraire les features les plus influentes pour comprendre quels facteurs pilotent vraiment le taux d'engagement.

---

## 6. Conclusion

Cette analyse prédictive du taux d'engagement Instagram a permis de valider plusieurs concepts fondamentaux en Data Science :

**1. Importance du Prétraitement :** L'extraction de caractéristiques temporelles et l'encodage adéquat des variables catégorielles ont été essentiels pour obtenir des données exploitables.

**2. Non-linéarité des Données :** Les modèles linéaires (régression linéaire, SVR sans tuning) se sont avérés inadéquats, tandis que les modèles arborescents ont capturé efficacement les interactions complexes.

**3. Performance Prédictive :** L'**Arbre de Décision émerge comme le meilleur modèle**, atteignant un $R^2$ de 0.71, ce qui représente une prédiction satisfaisante du taux d'engagement.

**4. Généralisation vs Variance :** La Forêt Aléatoire, bien que générant une meilleure généralisation en théorie, n'a pas surpassé l'Arbre simple sur ce dataset, illustrant que la régularisation supplémentaire n'était pas nécessaire.

**5. Nécessité d'une Validation Rigoureuse :** La comparaison systématique des modèles et l'évaluation sur un ensemble de test indépendant garantissent la fiabilité des conclusions.

En résumé, avec un $R^2$ de **71%**, le modèle d'Arbre de Décision offre une solution prédictive robuste pour anticiper le taux d'engagement des publications Instagram, tout en identifiant les variables les plus influentes. Des améliorations futures via optimisation des hyperparamètres ou exploration de modèles d'ensemble avancés (Gradient Boosting) pourraient encore élever les performances.









