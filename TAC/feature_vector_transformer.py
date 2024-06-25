import numpy as np
from sklearn.decomposition import PCA


class FeatureVectorTransformer:
    def __init__(self):
        self.j = None
        self.pca = None
        self.min_val = None
        self.max_val = None

    def closest_odd(self, k):
        possible_lower = int(np.floor(np.sqrt(k)))
        if possible_lower % 2 == 0:
            return possible_lower + 1
        else:
            return possible_lower

    def fit(self, feature_vectors):
        """
        Fit the transformer to the training dataset.

        :param feature_vectors: A matrix of training feature vectors.
        """
        n, k = feature_vectors.shape
        self.j = self.closest_odd(min(k, n))
        # Make sure n is smaller than j*j
        if n < self.j * self.j:
            self.j -= 2

        # Compute min and max values of the dataset for normalization
        self.min_val = np.min(feature_vectors)
        self.max_val = np.max(feature_vectors)

        if k > self.j * self.j:
            self.pca = PCA(n_components=self.j * self.j)
            self.pca.fit(feature_vectors)

    def normalize(self, vector):
        """
        Normalize a vector to the range [-1, 1] using the dataset's min and max values.

        :param vector: The input vector to be normalized.
        :return: The normalized vector.
        """
        normalized_vector = (
            2 * (vector - self.min_val) / (self.max_val - self.min_val) - 1
        )
        return normalized_vector

    def transform(self, feature_vectors):
        """
        Transform feature vectors to the closest odd kernel matrix.

        :param feature_vectors: A matrix of feature vectors to be transformed.
        :return: Transformed feature vectors as n matrices of jxj.
        """
        transformed_kernels = []

        for vector in feature_vectors:
            if len(vector) > self.j * self.j:
                # Reduce dimensions using PCA
                reduced_vector = self.pca.transform(vector.reshape(1, -1)).flatten()
            else:
                # Pad with zeros
                padding = np.zeros(self.j * self.j - len(vector))
                reduced_vector = np.concatenate((vector, padding))

            # Normalize the reduced/padded vector
            normalized_vector = self.normalize(reduced_vector)
            kernel_matrix = normalized_vector.reshape(self.j, self.j)
            transformed_kernels.append(kernel_matrix)

        return np.array(transformed_kernels)


if __name__ == "__main__":
    feature_vectors = np.random.rand(100, 27)
    transformer = FeatureVectorTransformer()
    transformer.fit(feature_vectors)
    matrix_kernels = transformer.transform(feature_vectors)
    print(matrix_kernels.shape)  # Should print (100, 5, 5)

    feature_vectors = np.random.rand(100, 8)
    transformer.fit(feature_vectors)
    matrix_kernels = transformer.transform(feature_vectors)
    print(matrix_kernels.shape)  # Should print (100, 3, 3)
