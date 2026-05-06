import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv('Dados_Projeto_Imagem_Ultrassom.xlsx - Planilha1.csv', decimal=',')

colunas_uteis = [
    'PESO', 'AC', 'AG', 'CC', 'AP', 'P.C', 'CT', 'CO', 
    'CCAB', 'LIL', 'LIS', 'Cga', 'Cper', 'PerPe', 'Ccau', 'DC'
]

df_final = df[colunas_uteis].copy().dropna()

X = df_final.drop(columns=['PESO'])
y = df_final['PESO']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=1)),
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(random_state=42))
])

param_grid = {
    'poly__degree': [1, 2],
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [None, 10, 20],
    'rf__min_samples_split': [2, 5, 10],
    'rf__max_features': ['sqrt', 'log2', None]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"Best Random Forest R2: {r2:.4f}")

