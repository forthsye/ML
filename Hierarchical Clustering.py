from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Load the dataset (Iris dataset)
iris = datasets.load_iris()
X = iris.data

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the Agglomerative Clustering model
agg_clust = AgglomerativeClustering(n_clusters=3)
agg_clust.fit(X_scaled)

# Evaluate model (using Silhouette Score)
sil_score = silhouette_score(X_scaled, agg_clust.labels_)
print(f"Silhouette Score: {sil_score:.4f}")
