import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import itertools


class LocalCluster(object):
    def __init__(self, config, kmeans_init):
        self.embedding_dimension = config.embedding_dimension
        self.windows_per_sample = config.windows_per_sample
        self.ft_bins = config.window_size // 2 + 1

        self.kmeans_init = kmeans_init
        self.n_init = 10
        self.N_assign = 0
        self.center = None

    def cluster2(self, embedding, sample_step, VAD_data_np):
        windows_per_sample = self.windows_per_sample
        embedding = embedding.reshape(windows_per_sample * self.ft_bins, self.embedding_dimension)


        k = 2

        kmeans = KMeans(k, random_state=0, n_init=self.n_init, init=self.kmeans_init)
        eg = kmeans.fit_predict(embedding)

        self.kmeans_init = kmeans.cluster_centers_
        self.n_init = 1

        mask = np.zeros((k, windows_per_sample * self.ft_bins))

        # TODO VAD:
        #for i in range(windows_per_sample):
        #    for j in range(self.ft_bins):
        #        if VAD_data_np[0, i, j] == 1:
        #           mask[i, j, kmean.labels_[ind]] = 1
        #            ind += 1

        for i in range(k):
            mask[i, eg == i] = 1

        mask = mask.reshape(k, windows_per_sample, self.ft_bins)

        return mask  # k, windows_per_sample, ft_bins
