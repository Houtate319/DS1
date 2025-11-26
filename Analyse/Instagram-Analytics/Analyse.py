# Import des librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Pour ignorer les warnings
import warnings
warnings.filterwarnings("ignore")

# Chargement du dataset
df = pd.read_csv('Instagram_Analytics.csv')

# Conversion de la colonne date en datetime
df['upload_date'] = pd.to_datetime(df['upload_date'])

# Extraction des caractéristiques temporelles
df['upload_month'] = df['upload_date'].dt.month
df['upload_day_of_week'] = df['upload_date'].dt.dayofweek  # Lundi=0, Dimanche=6
df['upload_hour'] = df['upload_date'].dt.hour
df['upload_year'] = df['upload_date'].dt.year

# Encodage One-Hot des variables catégorielles
categorical_cols = ['media_type', 'traffic_source', 'content_category']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Suppression des colonnes inutiles originelles
df_processed = df.drop(columns=['post_id', 'upload_date'])

# Vérification des valeurs manquantes
print("Valeurs manquantes par colonne :\n", df_processed.isnull().sum())

# Séparation des données en features X et cible y
y = df_processed['engagement_rate']
X = df_processed.drop(columns=['engagement_rate'])

# Séparation train/test (80%/20%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Import des modèles et métriques
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Régression Linéaire
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Régression Polynomiale (degré 2)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)
model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)
y_pred_poly = model_poly.predict(X_test_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)
r2_poly = r2_score(y_test, y_pred_poly)

# Arbre de Décision
model_dt = DecisionTreeRegressor(random_state=42)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
r2_dt = r2_score(y_test, y_pred_dt)

# Forêt Aléatoire
model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# SVR avec normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model_svr = SVR(kernel='rbf')
model_svr.fit(X_train_scaled, y_train)
y_pred_svr = model_svr.predict(X_test_scaled)
mse_svr = mean_squared_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mse_svr)
r2_svr = r2_score(y_test, y_pred_svr)

# Affichage des résultats
print(f"Régression Linéaire - R²: {r2_lr:.4f}, MSE: {mse_lr:.2f}, RMSE: {rmse_lr:.2f}")
print(f"Régression Polynomiale - R²: {r2_poly:.4f}, MSE: {mse_poly:.2f}, RMSE: {rmse_poly:.2f}")
print(f"Arbre de Décision - R²: {r2_dt:.4f}, MSE: {mse_dt:.2f}, RMSE: {rmse_dt:.2f}")
print(f"Forêt Aléatoire - R²: {r2_rf:.4f}, MSE: {mse_rf:.2f}, RMSE: {rmse_rf:.2f}")
print(f"SVR - R²: {r2_svr:.4f}, MSE: {mse_svr:.2f}, RMSE: {rmse_svr:.2f}")

# Visualisations

# Scatter plot : Réel vs Predits Arbre de Décision
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_dt, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Valeurs Réelles')
plt.ylabel('Valeurs Prédites')
plt.title('Arbre de Décision : Réel vs Prédit')
plt.show()

# Barplot des R2 et RMSE pour comparer modèles
metrics_df = pd.DataFrame({
    'Modèle': ['Régression Linéaire', 'Régression Polynomiale', 'Arbre de Décision', 'Forêt Aléatoire', 'SVR'],
    'R2': [r2_lr, r2_poly, r2_dt, r2_rf, r2_svr],
    'RMSE': [rmse_lr, rmse_poly, rmse_dt, rmse_rf, rmse_svr]
})
fig, ax1 = plt.subplots(figsize=(12,6))

color = 'tab:blue'
ax1.set_xlabel('Modèle')
ax1.set_ylabel('R2', color=color)
ax1.bar(metrics_df['Modèle'], metrics_df['R2'], color=color, alpha=0.6)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:green'
ax2.set_ylabel('RMSE', color=color)
ax2.plot(metrics_df['Modèle'], metrics_df['RMSE'], color=color, marker='o')
ax2.tick_params(axis='y', labelcolor=color)
ax2.invert_yaxis()  # Pour que RMSE faible soit en haut

plt.title('Comparaison des performances : R2 (barres) et RMSE (pointillés)')
plt.show()

