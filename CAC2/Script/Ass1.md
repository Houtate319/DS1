
<img src="HOUTATE Saïd CAC 2.jpg" style="height:464px;margin-right:432px"/>

# HOUTATE SAÏD

**Numéro d'étudiant** : 24010355  
**Classe** : CAC2

---

# Extrait de la base de données choisie

| InvoiceNo | StockCode | Description                         | Quantity | InvoiceDate         | UnitPrice | CustomerID | Country        | TotalPrice |
|-----------|-----------|-------------------------------------|----------|---------------------|-----------|------------|---------------|------------|
| 536365    | 85123A    | WHITE HANGING HEART T-LIGHT HOLDER | 6        | 2010-12-01 08:26:00 | 2.55      | 17850      | United Kingdom | 15.30      |
| 536365    | 71053     | WHITE METAL LANTERN                 | 6        | 2010-12-01 08:26:00 | 3.39      | 17850      | United Kingdom | 20.34      |
| ...       | ...       | ...                                 | ...      | ...                 | ...       | ...        | ...           | ...        |

> **Données extraites du jeu “Online Retail” de l’UCI Machine Learning Repository, nettoyées et formatées pour l’analyse.**

---

# Présentation de la base de données choisie : Online Retail (UCI)

La base de données "Online Retail" de l’UCI Machine Learning Repository est une célèbre source de données transactionnelles issues d’une entreprise britannique spécialisée dans la vente de cadeaux en ligne.

## Objectif et portée de la base

Cette base de données vise à fournir aux chercheurs et praticiens un ensemble réel et détaillé de transactions pour développer et tester des techniques analytiques, notamment en classification, en clustering, en découverte de patterns séquentiels, et en analyse RFM (Récence, Fréquence, Montant) appliquée au domaine du commerce électronique.  
Elle permet l’étude des comportements d’achat des clients et l’analyse des schémas transactionnels pour améliorer le marketing, la segmentation, ou la détection de fraudes.

## Créateurs et laboratoire

La base a été créée par Daqing Chen, avec la participation de Sai Laing Sain et Kun Guo.  
Les auteurs sont affiliés à la School of Engineering, London South Bank University au Royaume-Uni.  
Le contact principal est [chend@lsbu.ac.uk](mailto:chend@lsbu.ac.uk).

## Année et publication

Les données couvrent la période du 1er décembre 2010 au 9 décembre 2011.  
Le jeu de données fut officiellement publié en 2015 sur le repository de l’UCI Machine Learning, avec des analyses et applications citées dès 2012 dans des revues académiques telles que le "Journal of Database Marketing and Customer Strategy Management".  
Le DOI officiel est 10.24432/C5BW33.

## Pourquoi et contexte d’utilisation

Le but de cette base de données est d’offrir un référentiel fiable aux universitaires et data scientists pour :
- Analyser les comportements d’achat et de retour produits (avec les annulations de factures identifiées par un 'C' dans le champ InvoiceNo)
- Développer des modèles de segmentation client ou de détection de fraudes
- Expérimenter sur de vraies transactions tirées d’un contexte commercial international auprès de détaillants comme de grossistes.
- Appliquer et tester divers
  
---

# Etapes d'analyse :

1. Environnement Python et Installation
Installation des bibliothèques essentielles:
```
!pip install pandas matplotlib seaborn ucimlrepo
```
---
2. Importation et Chargement du Dataset
Importation, chargement et aperçu:
```
from ucimlrepo import fetch_ucirepo
import pandas as pd

# Télécharger et charger la base Online Retail
online_retail = fetch_ucirepo(id=352)
df = online_retail.data.features

# Afficher les 5 premières lignes
print(df.head())
print(df.info())
```
Sortie typique : un tableau de données brutes avec les colonnes principales : InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country.

---
3. Nettoyage et Préparation des Données
Étapes du nettoyage :
```
df = df.dropna(subset=["Description", "CustomerID"])
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]
print(f"Dimensions après nettoyage : {df.shape}")
```
On retire les lignes avec valeurs manquantes, quantités/prix non-positifs (retours, erreurs).

---
4. Création de nouvelles variables
Ajout du total de vente et variables temporelles :
```
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Year"] = df["InvoiceDate"].dt.year
df["Month"] = df["InvoiceDate"].dt.month
df["Day"] = df["InvoiceDate"].dt.day
df["Hour"] = df["InvoiceDate"].dt.hour
print(df[["InvoiceNo", "Quantity", "UnitPrice", "TotalPrice", "Year", "Month", "Day", "Hour"]].head())
```
Cela permet l’analyse temporelle fine et le calcul agrégé.

---
5. Analyse des tendances de vente
Ventes quotidiennes et mensuelles :
```
# Quotidiennes
dailysales = df.groupby(["Year", "Month", "Day"])["TotalPrice"].sum().reset_index()
print(dailysales.head())

# Mensuelles
monthlysales = df.groupby(["Year", "Month"])["TotalPrice"].sum().reset_index()
print(monthlysales.head())
```
On détecte les pics d’activité, jours/mois record affichés.

---
6. Segmentation produits et clients
Top produits vendus et rentabilité :
```
topselling = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
print(topselling)

topprofit = df.groupby("Description")["TotalPrice"].sum().sort_values(ascending=False).head(10)
print(topprofit)
```
On obtient la liste des best-sellers et des produits générant le plus de chiffre d’affaires.

Top clients actifs et rentables :
```
top_customers = df.groupby("CustomerID")["InvoiceNo"].nunique().sort_values(ascending=False).head(10)
print(top_customers)

top_profit_clients = df.groupby("CustomerID")["TotalPrice"].sum().sort_values(ascending=False).head(10)
print(top_profit_clients)

```
Identification des clients les plus fidèles et les plus profitables.

---
7. Analyse géographique des ventes
Répartition par pays :
```
sales_by_country = df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False)
print(sales_by_country.head(10))
```
On distingue clairement les marchés majeurs de l’entreprise.

---
8. Visualisation des résultats
Exemples graphiques (tendances, produits, clients, géographie) :
```
import matplotlib.pyplot as plt
import seaborn as sns

# Ventes mensuelles
monthlysales["YearMonth"] = pd.to_datetime(monthlysales["Year"].astype(str) + "-" + monthlysales["Month"].astype(str))
plt.figure(figsize=(12,6))
sns.lineplot(x="YearMonth", y="TotalPrice", data=monthlysales, marker="o")
plt.title("Tendances des ventes mensuelles")
plt.xlabel("Mois")
plt.ylabel("Ventes mensuelles")
plt.xticks(rotation=45)
plt.show()

# Top produits rentables
plt.figure(figsize=(12,6))
topprofit.plot(kind="bar")
plt.title("Top produits rentables")
plt.ylabel("Chiffre d'affaires (£)")
plt.show()

# Top pays
plt.figure(figsize=(12,6))
sales_by_country.head(10).plot(kind="bar")
plt.title("Top 10 Pays par CA")
plt.ylabel("Ventes totales (£)")
plt.xticks(rotation=45)
plt.show()
```
Graphiques lisibles, légendés, illustrant les principaux enseignements.

---
9. Synthèse et Commentaires
La base nettoyée comporte ~398 000 transactions valides.

Les produits best-sellers et top clients sont rigoureusement identifiés.

Les périodes de haute activité : décembre 2011 (jour record), novembre 2011 (mois record).

Les marchés majeurs sont le Royaume-Uni, les Pays-Bas, l’Irlande et l’Allemagne.

Tous les résultats sont visualisés et expliqués.





