import numpy as np

from tqdm import tqdm


class MahalanobisClassifier:
    def __init__(self, num_class):
        self.num_class = num_class
        self.class_means = {}
        self.cov_matrices = {}

    def train(self, x, y):
        for class_label in tqdm(range(self.num_class)):
            class_samples = x[y == class_label]
            self.class_means[class_label] = np.mean(class_samples, axis=0)
            self.cov_matrices[class_label] = np.cov(class_samples, rowvar=False)

    def test(self, x, y):
        total = x.shape[0]

        mahalanobis_distances = np.zeros((len(x), self.num_class))

        for class_label, class_mean in tqdm(self.class_means.items()):
            cov_matrix = self.cov_matrices[class_label]
            x_minus_mean = x - class_mean
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            mahalanobis = np.sum(np.dot(np.dot(x_minus_mean, inv_cov_matrix), x_minus_mean.T), axis=1)
            mahalanobis_distances[:, class_label] = mahalanobis

        predicted_labels = np.argmin(mahalanobis_distances, axis=1)
        correct = (np.equal(predicted_labels, y)).sum().item()
        accuracy = 100 * correct / total

        return accuracy
