import numpy as np
import faiss


class KNN:
    def __init__(self, k=5):
        self.index = None
        self.clf_y = None
        self.gt_clf_y = None
        self.y = None
        self.gt_y = None
        self.dataset_idx = None
        self.k = k + 1 # b/c faiss returns itself as well.

    def fit(self, X, clf_y, gt_clf_y, y, gt_y, idx, dist_metric='l2', normalize_feature=True):
        if dist_metric == 'l2':
            self.index = faiss.IndexFlatL2(X.shape[1])
        elif dist_metric == 'cosine':
            self.index = faiss.IndexFlatIP(X.shape[1])
            if normalize_feature:
                faiss.normalize_L2(X)
        else:
            NotImplementedError

        self.index.add(x=X.astype(np.float32)) # has to be float32.
        self.clf_y = np.array(clf_y)
        self.gt_clf_y = np.array(gt_clf_y)
        self.y = np.array(y)
        self.gt_y = np.array(gt_y)
        self.dataset_idx = np.array(idx)

    def query(self, X, q_i, query_in_neighbor=True, k=None):
        if k==None:
            k = self.k
        distances, indices = self.index.search(x=X.astype(np.float32), k=k)

        if query_in_neighbor:
            indices = indices[:, 1:]
            distances = distances[:, 1:]
            n_meta = self.y[indices], self.gt_y[indices], self.clf_y[indices], self.gt_clf_y[indices], self.dataset_idx[indices]
            q_meta = self.y[q_i], self.gt_y[q_i], self.clf_y[q_i], self.gt_clf_y[q_i], self.dataset_idx[q_i]
            return distances, indices, q_meta, n_meta

        else:
            n_meta = self.y[indices], self.gt_y[indices], self.clf_y[indices], self.gt_clf_y[indices], self.dataset_idx[indices]
            return distances, indices, n_meta


    def index_cpu_to_gpu(self):
        self.res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index)
