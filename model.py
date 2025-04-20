import numpy as np

class KMeans:
    def __init__(self, k: int, epsilon: float = 1e-6) -> None:
        self.num_clusters = k
        self.cluster_centers = None
        self.epsilon = epsilon
    
    def fit(self, X: np.ndarray, max_iter: int = 100) -> None:
        # Randomly initialize cluster centers by selecting k unique points from X
        np.random.seed(42)  # For reproducibility
        random_indices = np.random.choice(X.shape[0], self.num_clusters, replace=False)
        self.cluster_centers = X[random_indices]
    
        for _ in range(max_iter):
            # Assign each sample to the closest cluster center
            distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers, axis=2)
            labels = np.argmin(distances, axis=1)
    
            # Compute new cluster centers as the mean of assigned points
            new_cluster_centers = np.array([
                X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 else self.cluster_centers[i]
                for i in range(self.num_clusters)
            ])
    
            # Check for convergence (if the cluster centers do not change significantly)
            if np.linalg.norm(new_cluster_centers - self.cluster_centers) < self.epsilon:
                break
    
            self.cluster_centers = new_cluster_centers
        # Initialize cluster centers (need to be careful with the initialization,
        # otherwise you might see that none of the pixels are assigned to some
        # of the clusters, which will result in a division by zero error)
        
        

        
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Predicts the index of the closest cluster center for each data point
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers, axis=2)
        labels = np.argmin(distances, axis=1)
        #print("Predict labels shape:", labels.shape)
        return labels
        #raise NotImplementedError
    
    def fit_predict(self, X: np.ndarray, max_iter: int = 100) -> np.ndarray:
        self.fit(X, max_iter)
        return self.predict(X)
    
    def replace_with_cluster_centers(self, X: np.ndarray) -> np.ndarray:
        # Returns an ndarray of the same shape as X
        # Each row of the output is the cluster center closest to the corresponding row in X
        labels = self.predict(X)  # Get the cluster labels for each point
        #print("Labels shape:", labels.shape)
        replaced_X = self.cluster_centers[labels]  # Replace each point with its cluster center
        return replaced_X