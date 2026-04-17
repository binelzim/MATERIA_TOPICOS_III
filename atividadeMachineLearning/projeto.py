import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv('heart_disease_dataset.csv')


variaveis_categoricas = ['Gender', 'Smoking', 'Alcohol Intake', 'Family History', 
                         'Diabetes', 'Obesity', 'Exercise Induced Angina', 'Chest Pain Type']

df_transformado = pd.get_dummies(df, columns=variaveis_categoricas, drop_first=True)

X = df_transformado.drop('Heart Disease', axis=1)
y = df_transformado['Heart Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
log_reg = LogisticRegression(max_iter=1000)

knn.fit(X_train_scaled, y_train)
log_reg.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)
y_pred_log = log_reg.predict(X_test_scaled)

print("--- Métricas do KNN ---")
print(classification_report(y_test, y_pred_knn))
print("Acurácia KNN:", accuracy_score(y_test, y_pred_knn))

print("\n--- Métricas da Regressão Logística ---")
print(classification_report(y_test, y_pred_log))
print("Acurácia Regressão Logística:", accuracy_score(y_test, y_pred_log))

importancias = pd.DataFrame({
    'Característica': X.columns,
    'Peso (Coeficiente)': log_reg.coef_[0]
})
print("\n--- Pesos das Características (Regressão Logística) ---")
print(importancias.reindex(importancias['Peso (Coeficiente)'].abs().sort_values(ascending=False).index))