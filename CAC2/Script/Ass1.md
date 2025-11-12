
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







