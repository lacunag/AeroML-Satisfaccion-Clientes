import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.decomposition import PCA

data = pd.read_excel('satisfaccion_AeroML.xlsx', engine='openpyxl', index_col=0)

print(data.info())
print(data.describe())
print(data.head())

# Manejo de valores faltantes (ejemplo, eliminar filas con valores nulos)
data.dropna(inplace=True)

# Codificación de variables categóricas (ejemplo, one-hot encoding)
data = pd.get_dummies(data, columns=['genero', 'tipo_cliente', 'tipo_viaje', 'clase'])

# Normalización/estandarización (ejemplo, Min-Max scaling)

scaler = MinMaxScaler()
data[['edad', 'distancia']] = scaler.fit_transform(data[['edad', 'distancia']])

# Selección de características relevantes
selected_features = ['edad', 'distancia', 'servicio_wifi', 'comodidad_asientos', 'entretenimiento', 'limpieza', 'satisfaccion']
data = data[selected_features]

# División del conjunto de datos
X = data.drop('satisfaccion', axis=1)
y = data['satisfaccion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Elección del modelo: Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenamiento del modelo
model.fit(X_train, y_train)

# Predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Cálculo de métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='satisfied')
recall = recall_score(y_test, y_pred, pos_label='satisfied')
f1 = f1_score(y_test, y_pred, pos_label='satisfied')

# Impresión de métricas
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicciones')
plt.ylabel('Valores reales')
plt.title('Matriz de Confusión')
plt.show()

# Para Random Forest, podemos obtener información sobre la importancia de las características
feature_importance = model.feature_importances_

# Visualizar la importancia de las características
plt.barh(X.columns, feature_importance)
plt.xlabel('Importancia')
plt.title('Importancia de las Características')
plt.show()

# Analisis de componentes
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Convertir etiquetas de satisfacción a colores
color_map = {'neutral or dissatisfied': 'red', 'satisfied': 'green'}  # Puedes agregar más colores según tus necesidades
colors = y.map(color_map)

# Visualizar los resultados de PCA
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

