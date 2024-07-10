import numpy as np
from sklearn.decomposition import PCA


class FeatureVectorTransformer:
    def __init__(self, max_kernel_size=11, scale_factor=1.0):
        self.j = None
        self.pca = None
        self.min_val = None
        self.max_val = None
        self.max_kernel_size = max_kernel_size
        self.scale_factor = scale_factor

    def closest_odd(self, n, k):
        possible_lower = int(np.floor(np.sqrt(k)))
        # Ensure j is odd
        possible_lower += 1 - possible_lower % 2
        # Be able to use PCA
        while possible_lower * possible_lower > n:
            possible_lower -= 2
        # Make sure j is not larger than max_kernel_size
        possible_lower = min(possible_lower, self.max_kernel_size)
        return possible_lower

    def fit(self, feature_vectors):
        """
        Fit the transformer to the training dataset.

        :param feature_vectors: A matrix of training feature vectors.
        """
        n, k = feature_vectors.shape
        self.j = self.closest_odd(n, k)

        # Compute min and max values of the dataset for normalization
        self.min_val = np.min(feature_vectors)
        self.max_val = np.max(feature_vectors)

        self.upsample = True
        if k > self.j * self.j:
            # k is larger than j * j, we reduce dimensions
            self.upsample = False
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
        return normalized_vector.clip(-1, 1)

    def transform(self, feature_vectors):
        """
        Transform feature vectors to the closest odd kernel matrix.

        :param feature_vectors: A matrix of feature vectors to be transformed.
        :return: Transformed feature vectors as n matrices of jxj.
        """
        transformed_kernels = []

        for vector in feature_vectors:
            # Min-max normalization from [-1, 1]
            normalized_vector = self.normalize(vector)
            if self.upsample:
                # Pad with zeros
                padding = np.zeros(self.j * self.j - len(normalized_vector))
                normalized_vector = np.concatenate((normalized_vector, padding))
            else:
                # Reduce dimensions using PCA
                normalized_vector = self.pca.transform(
                    normalized_vector.reshape(1, -1)
                ).flatten()
            kernel_matrix = (
                normalized_vector.reshape(self.j, self.j) * self.scale_factor
            )
            transformed_kernels.append(kernel_matrix)

        return np.array(transformed_kernels)


if __name__ == "__main__":
    feature_vectors = np.random.rand(100, 500)
    transformer = FeatureVectorTransformer()
    transformer.fit(feature_vectors)
    matrix_kernels = transformer.transform(feature_vectors)
    print(matrix_kernels.shape)  # Should print (100, 9, 9)

    feature_vectors = np.random.rand(100, 8)
    transformer.fit(feature_vectors)
    matrix_kernels = transformer.transform(feature_vectors)
    print(matrix_kernels.shape)  # Should print (100, 3, 3)

    feature_vectors = np.random.rand(100, 10)
    transformer.fit(feature_vectors)
    matrix_kernels = transformer.transform(feature_vectors)
    print(matrix_kernels.shape)  # Should print (100, 3, 3)

    feature_vectors = np.random.rand(100, 17)
    transformer.fit(feature_vectors)
    matrix_kernels = transformer.transform(feature_vectors)
    print(matrix_kernels.shape)  # Should print (100, 5, 5)
