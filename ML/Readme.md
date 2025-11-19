<img src="HOUTATE Saïd CAC 2.jpg" style="height:300px;margin-right:300px; float:left; border-radius:10px;"/>

# HOUTATE SAÏD

**Numéro d'étudiant** : 24010355  
**Classe** : CAC2

<br clear="left"/>

---

# Compte rendu
## Analyse de la Qualité du Vin et Classification par k-NN

**Date :** 19 Novembre 2025

---

## Table des Matières

1.  [Introduction et Contexte](#1-introduction-et-contexte)
2.  [Analyse Exploratoire des Données (Data Analysis)](#2-analyse-exploratoire-des-données-data-analysis)
    * [Chargement et Structure du Dataset](#21-chargement-et-structure-du-dataset)
    * [Distribution de la Variable Cible](#22-distribution-de-la-variable-cible)
    * [Transformation en Problème de Classification Binaire](#23-transformation-en-problème-de-classification-binaire)
    * [Analyse Statistique et Visuelle](#24-analyse-statistique-et-visuelle)
3.  [Méthodologie de Classification](#3-méthodologie-de-classification)
    * [Séparation des Données (Data Split)](#31-séparation-des-données-data-split)
    * [Algorithme k-NN (k-Nearest Neighbors)](#32-algorithme-k-nn-k-nearest-neighbors)
4.  [Résultats et Impact de la Normalisation](#4-résultats-et-impact-de-la-normalisation)
    * [Données Brutes (Non normalisées)](#41-données-brutes-non-normalisées)
    * [Données Normalisées](#42-données-normalisées)
    * [Comparaison des Performances](#43-comparaison-des-performances)
5.  [Conclusion](#5-conclusion)

---

## 1. Introduction et Contexte

Ce rapport présente une analyse détaillée d'un jeu de données réel concernant la qualité du vin blanc, réalisée dans le cadre du cours de Science des Données. En suivant le cycle de vie des données, nous avons mené une exploration (EDA), un prétraitement et une modélisation prédictive.

L'objectif est de construire un modèle de classification capable de prédire si un vin est de "bonne" ou "mauvaise" qualité en utilisant l'algorithme des **k-Plus Proches Voisins (k-NN)**, et d'évaluer l'impact critique de la normalisation des données sur la précision du modèle.

---

## 2. Analyse Exploratoire des Données (Data Analysis)

### 2.1 Chargement et Structure du Dataset

Le jeu de données `winequality-white.csv` contient les propriétés physico-chimiques de divers vins blancs portugais "Vinho Verde".

* **Nombre d'échantillons ($N$) :** 4898 observations.
* **Nombre de variables ($d$) :** 12 colonnes (11 features + 1 target).

**Variables d'entrée ($X$) :** Acidité (fixe, volatile), acide citrique, sucre résiduel, chlorures, soufre (libre, total), densité, pH, sulfates, alcool.  
**Variable de sortie ($Y$) :** Note de qualité (`quality`) entre 0 et 10.

```python
import pandas as pd
# Chargement des données avec le séparateur correct
link = "winequality-white.csv"
df = pd.read_csv(link, header="infer", delimiter=";")

print("========= Résumé du Dataset =========")
df.info()
print("\n========= Premiers échantillons =========")
print(df.head())
```
### 2.2 Distribution de la Variable Cible
L'analyse de la colonne `quality` montre une distribution inégale, avec une majorité de vins notés entre 5 et 7.
| Qualité (Note) | Nombre d'échantillons |
|----------------|-----------------------|
|6               |2198                   |
|5               |1457                   |
|7               |880                    |
|8               |175                    |
|4               |163                    |
|3               |20                     |
|9               |5                      |

### 2.3 Transformation en Problème de Classification Binaire
Pour simplifier la modélisation, nous avons transformé le problème de régression en classification binaire :

* **Classe 0 (Mauvais vin) :** $Quality \le 5$
* **Classe 1 (Bon vin) :** $Quality > 5$

```python
# Création des vecteurs X et Y
X = df.drop("quality", axis=1)
Y = df["quality"]

# Binarisation de la cible
Y = [0 if val <= 5 else 1 for val in Y]
```
### 2.4 Analyse Statistique et Visuelle
L'analyse des corrélations et la visualisation par **Boxplots** ont révélé des différences d'échelle importantes entre les variables. Par exemple, le `total sulfur dioxide` a des valeurs bien supérieures aux `chlorides` ou au `pH`.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Boxplot pour visualiser les échelles et outliers
plt.figure(figsize=(12, 6))
ax = plt.gca()
sns.boxplot(data=X, orient="v", palette="Set1", width=0.5, notch=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.title("Distribution des variables (Données brutes)")
plt.show()

# Matrice de corrélation
plt.figure(figsize=(10, 8))
corr = X.corr()
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title("Matrice de Corrélation")
plt.show()
```
(Les graphiques générés par ce code montrent clairement la nécessité de normaliser les données pour que le k-NN ne soit pas biaisé par les variables à grande échelle.)

---
# 3. Méthodologie de Classification
## 3.1 Séparation des Données (Data Split)
Pour évaluer la capacité de généralisation du modèle, nous avons divisé les données en trois ensembles :

 **1.** Entraînement ($D_a$) : Pour entraîner le modèle.
 **2.** Validation ($D_v$) : Pour régler l'hyperparamètre $k$.
 **3.** Test ($D_t$) : Pour l'évaluation finale de la performance.

L'option `stratify=Y` garantit que la proportion de bons et mauvais vins est conservée dans chaque sous-ensemble.

```python
from sklearn.model_selection import train_test_split

# Séparation Test (1/3)
Xa, Xt, Ya, Yt = train_test_split(X, Y, shuffle=True, test_size=1/3, stratify=Y)
# Séparation Train/Validation (50/50 du reste)
Xa, Xv, Ya, Yv = train_test_split(Xa, Ya, shuffle=True, test_size=0.5, stratify=Ya)
```
## 3.2 Algorithme k-NN (k-Nearest Neighbors)
Nous avons implémenté le k-NN et testé différentes valeurs de $k$ (nombre de voisins) pour trouver l'optimum.

* **k petit (ex: k=1) :** Risque de sur-apprentissage (Overfitting), le modèle est trop sensible au bruit.
* **k grand :** Risque de sous-apprentissage (Underfitting), le modèle lisse trop la frontière de décision.

  ---
# 4. Résultats et Impact de la Normalisation
## 4.1 Données Brutes (Non normalisées)
Sur les données brutes, le modèle k-NN est performant médiocre car la distance euclidienne est dominée par les variables à grande échelle (comme le soufre total).

* **Meilleur $k$ trouvé :** 1 (Indice de sur-apprentissage)
* **Erreur de validation :** $\approx 31.17\%$
* **Erreur de test :** $\approx 32.15\%$

## 4.2 Données Normalisées
Nous avons appliqué une standardisation (`StandardScaler`) pour centrer et réduire les variables ($moyenne=0, variance=1$).Important : Le scaler est ajusté (`fit`) uniquement sur l'ensemble d'entraînement, puis appliqué (`transform`) sur les ensembles de validation et de test pour éviter la fuite de données (data leakage).

```python
from sklearn.preprocessing import StandardScaler

# Normalisation
sc = StandardScaler()
sc.fit(Xa) # Fit uniquement sur le train
Xa_n = sc.transform(Xa)
Xv_n = sc.transform(Xv)
Xt_n = sc.transform(Xt)
```

## 4.3 Comparaison des Performances
L'impact de la normalisation est spectaculaire, comme le montre le tableau ci-dessous :

|Méthode            |Meilleur k|Erreur Validation|Erreur Test   |Performance        |
|-------------------|----------|-----------------|--------------|-------------------|
|Données Brutes     |1         |0.3117 (31.2%)   |0.3215 (32.2%)|Faible             |
|Données Normalisées|17        |0.2370 (23.7%)   |0.2529 (25.3%)|Nettement Améliorée|

Le meilleur $k$ passe de 1 à 17, indiquant un modèle plus robuste et généralisable. L'erreur sur le jeu de test diminue d'environ 7 points de pourcentage.

---
# 5. Conclusion
Ce TP a permis de valider plusieurs concepts clés en Data Science :
**1. Exploration :** Comprendre la distribution des données est crucial avant toute modélisation.
**2. Prétraitement :** La normalisation est une étape indispensable pour les algorithmes basés sur la distance comme le k-NN. Sans elle, les résultats sont biaisés et sous-optimaux.
**3. Méthodologie :** La séparation rigoureuse des données (Train/Val/Test) et l'utilisation de la validation pour choisir les hyperparamètres ($k$) permettent de construire des modèles fiables et d'éviter le sur-apprentissage.
