import numpy as np
import pickle

from torchvision.datasets import FashionMNIST
from torchvision.transforms import transforms
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class CIFARDataset:
    def __init__(self, pca=False, lda=False, pca_dim=None, lda_dim=None):
        images = []
        labels = []

        train_path = "./Dataset/cifar-10-batches-py/data_batch_{}"
        test_path = "./Dataset/cifar-10-batches-py/test_batch"
        for i in range(1, 6):
            x, y = load_file(train_path.format(i))
            images.append(x)
            labels.append(y)

        train_split = 50000

        x, y = load_file(test_path)
        images.append(x)
        labels.append(y)

        images = np.concatenate(images)
        labels = np.concatenate(labels)
        self.num_class = 10

        std = np.std(images, axis=0)
        mean = np.mean(images, axis=0)

        images = (images - mean) / std
        images = (images - np.min(images, axis=0)) / (np.max(images, axis=0) - np.min(images, axis=0)) * 255

        if pca:
            pca = PCA(n_components=pca_dim)
            images = pca.fit_transform(images)

        self.train_x = images[:train_split]
        self.train_y = labels[:train_split]

        self.test_x = images[train_split:]
        self.test_y = labels[train_split:]

        if lda:
            lda = LinearDiscriminantAnalysis(n_components=lda_dim)
            self.train_x = lda.fit_transform(self.train_x, self.train_y)
            self.test_x = lda.transform(self.test_x)


def load_file(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        x, y = data[b'data'], data[b'labels']
    return x, y


class FashionMNISTDataset:
    def __init__(self, pca=False, lda=False, pca_dim=None, lda_dim=None):
        train_data = FashionMNIST('./Dataset', train=True, download=False, transform=transforms.ToTensor())
        test_data = FashionMNIST('./Dataset', train=False, download=False, transform=transforms.ToTensor())
        data = train_data + test_data
        train_split = len(train_data)
        self.num_class = 10

        images = []
        labels = []

        for image, label in data:
            images.append(image.view(-1).numpy())
            labels.append(label)

        images = np.array(images)
        labels = np.array(labels)

        std = np.std(images, axis=0)
        mean = np.mean(images, axis=0)

        images = (images - mean) / std
        images = (images - np.min(images, axis=0)) / (np.max(images, axis=0) - np.min(images, axis=0)) * 255

        if pca:
            pca = PCA(n_components=pca_dim)
            images = pca.fit_transform(images)

        self.train_x = images[:train_split]
        self.train_y = labels[:train_split]

        self.test_x = images[train_split:]
        self.test_y = labels[train_split:]

        if lda:
            lda = LinearDiscriminantAnalysis(n_components=lda_dim)
            self.train_x = lda.fit_transform(self.train_x, self.train_y)
            self.test_x = lda.transform(self.test_x)

# d = FashionMNISTDataset()
