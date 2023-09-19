import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate a synthetic dataset with two features
np.random.seed(0)
data = np.random.randn(100, 2)  # 100 data points with 2 features

# Create a PCA instance
pca = PCA(n_components=2)

# Fit the PCA model to the data
pca.fit(data)

# Transform the data into the PCA space
data_pca = pca.transform(data)

# Plot the original data
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1])
plt.title('Original Data')

# Plot the data in PCA space
plt.subplot(1, 2, 2)
plt.scatter(data_pca[:, 0], data_pca[:, 1])
plt.title('Data in PCA Space')

# Plot the principal components as vectors
for i, (variance, vector) in enumerate(zip(pca.explained_variance_ratio_, pca.components_)):
    plt.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=3 * np.sqrt(variance), color=f'C{i}', label=f'PC{i + 1}')

plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.legend()
plt.title('Principal Components')

# Display the plots
plt.show()
