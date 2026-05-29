import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('credito.csv')

numeric_features = ['Age', 'Debt', 'YearsEmployed', 'CreditScore', 'ZipCode', 'Income']
X = df[numeric_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
K_range = range(2, 9)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o', linestyle='--')
plt.title('Método do Cotovelo (Elbow Method)')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia')
plt.grid(True)
plt.savefig('cotovelo.png')
plt.close()

kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(X_scaled)

print("\n--- PERFIL GERAL (MÉDIAS) ---")
print(df.groupby('Cluster')[numeric_features].mean())

print("\n--- RENDA MÉDIA DE CADA GRUPO ---")
print(df.groupby('Cluster')['Income'].mean())

print("\n--- SETOR (INDUSTRY) PREDOMINANTE POR GRUPO ---")
for c in range(3):
    mode_industry = df[df['Cluster'] == c]['Industry'].mode().iloc[0]
    print(f"Cluster {c}: {mode_industry}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame({'PCA1': X_pca[:, 0], 'PCA2': X_pca[:, 1], 'Cluster': df['Cluster']})

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', palette='Set1', data=df_pca, s=100)
plt.title('Separação dos Grupos (Visualização 2D com PCA)')
plt.savefig('pca_clusters.png')
plt.close()