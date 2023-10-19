from Classifier import MahalanobisClassifier
from Data import CIFARDataset, FashionMNISTDataset


pca_dim = 600
lda_dim = 8

dataset = FashionMNISTDataset(pca=True, lda=True, pca_dim=pca_dim, lda_dim=lda_dim)

train_x, train_y = dataset.train_x, dataset.train_y
test_x, test_y = dataset.test_x, dataset.test_y

classifier = MahalanobisClassifier(dataset.num_class)

classifier.train(train_x, train_y)

accuracy = classifier.test(test_x, test_y)
s = "PCA(n = {}) LDA(n = {})".format(pca_dim, lda_dim) + str(accuracy)

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
