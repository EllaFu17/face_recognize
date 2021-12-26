import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

def gmm_pca(train, d):
    pca1 = PCA(n_components = d)
    train = pca1.fit_transform(train)

    gmm = GaussianMixture(n_components=3)
    gmm.fit(train)
    pred = gmm.predict(train)
    # comp = gmm.predict_proba(train) # （4260，3）
    # component = np.where(comp == np.max(comp, axis = 1))
    # print(component)

    # visualize
    pca = PCA(n_components=2)
    reduced_train = pca.fit_transform(train)
    plt.figure()
    plt.scatter(reduced_train[:, 0], reduced_train[:, 1], s = 2, c = pred)
    plt.title('GMM visualization, D = ' + str(d))

dataset = scipy.io.loadmat('/Users/fuyalun/Documents/EE5907-PatternRecognize/CA2/CA2-FuYalun/facedata.mat')
train_data = np.array(dataset[ "train_data" ])
test_data = np.array(dataset[ "test_data" ])
test_label = np.array(dataset[ "test_label" ])
train_label = np.array(dataset[ "train_label" ])
print(train_label.shape, test_label.shape)

train_all = np.transpose(np.hstack((train_data, test_data))) # (4260, 1024)
print(train_all.shape)

gmm_pca(train_all, 80)
gmm_pca(train_all, 200)
gmm_pca(train_all, 1024)
plt.show()  
