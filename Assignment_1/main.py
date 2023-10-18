import numpy as np
import torch

from PIL import Image
from torchvision.transforms import transforms
from torchvision.models import resnet50, ResNet50_Weights

from Classifier import MahalanobisClassifier
from Data import CIFARDataset, FashionMNISTDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
model.eval()

dataset = CIFARDataset()

train_x, train_y = dataset.train_x, dataset.train_y
test_x, test_y = dataset.test_x, dataset.test_y

dim = 1

dataset = FashionMNISTDataset(lda=True, lda_dim=dim)

classifier = MahalanobisClassifier(dataset.num_class)

classifier.train(train_x, train_y)

accuracy = classifier.test(test_x, test_y)
s = "LDA(n = {})".format(dim) + str(accuracy)

print(s)

# with open("./result_CIFAR-10.txt", 'w') as f:
#     # ls = [2] + list(range(5, 100, 5)) + list(range(100, 700, 100))
#     # for i in ls:
#     #     dataset = CIFARDataset(pca=True, pca_dim=i)
#     #
#     #     train_x, train_y = dataset.train_x, dataset.train_y
#     #     test_x, test_y = dataset.test_x, dataset.test_y
#     #
#     #     classifier = MahalanobisClassifier(dataset.num_class)
#     #
#     #     classifier.train(train_x, train_y)
#     #
#     #     accuracy = classifier.test(test_x, test_y)
#     #     s = "PCA(n = {})".format(i) + str(accuracy)
#     #     f.write(s + '\n')
#     #
#     #     print(s)
#
#     for i in range(2, 10):
#         dataset = FashionMNISTDataset(lda=True, lda_dim=i)
#
#         train_x, train_y = dataset.train_x, dataset.train_y
#         test_x, test_y = dataset.test_x, dataset.test_y
#
#         classifier = MahalanobisClassifier(dataset.num_class)
#
#         classifier.train(train_x, train_y)
#
#         accuracy = classifier.test(test_x, test_y)
#         s = "LDA(n = {})".format(i) + str(accuracy)
#
#         print(s)

# total = test_dataset.__len__()
# correct = 0
# for images, labels in tqdm(test_loader):
#     with torch.no_grad():
#         images = images.to(device)
#         labels = labels.numpy()
#         visual_features = model(images).cpu().numpy()
#         classifier.get_test_data(visual_features, labels)
#
# accuracy = classifier.test()
#
# print(accuracy)
