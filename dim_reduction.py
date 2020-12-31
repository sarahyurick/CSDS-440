import numpy as np
from sklearn.decomposition import PCA, SparsePCA, TruncatedSVD, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy import linalg


def perform_pca(dataset, method):
    """
    Research extension implementation: Versions of Principle Components Analysis.
    """
    dataset = np.array(dataset)

    if method == "pca":
        dataset = StandardScaler().fit_transform(dataset)
        # see https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        pca = PCA(n_components=0.95)
        pca_data = pca.fit_transform(dataset)
    elif method == "sparse":
        # see https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html
        transformer = SparsePCA(n_components=10, random_state=0)
        transformer.fit(dataset)
        pca_data = transformer.transform(dataset)
    elif method == "truncated":
        # see https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
        svd = TruncatedSVD(n_components=10, n_iter=7, random_state=42)
        svd.fit(dataset)
        pca_data = svd.transform(dataset)
    elif method == "kernel":
        # see https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html
        transformer = KernelPCA(n_components=10, eigen_solver='arpack')
        pca_data = transformer.fit_transform(dataset)
    else:
        raise RuntimeError("Must enter a valid PCA method")

    return pca_data


def perform_pca_double(dataset1, dataset2, method):
    """
    Research extension 2. Versions of PCA,
    this time applying the dimensions fitted to dataset1 (the auxiliary dataset)
    to BOTH dataset1 and dataset2 (the target dataset),
    instead of fitting both dataset1 and dataset2 separately.
    """
    dataset1 = np.array(dataset1)
    dataset2 = np.array(dataset2)

    if method == "pca_double":
        dataset1 = StandardScaler().fit_transform(dataset1)
        dataset2 = StandardScaler().fit_transform(dataset2)
        pca = PCA(n_components=0.95)
        pca.fit(dataset1)
        pca_data1 = pca.transform(dataset1)
        pca_data2 = pca.transform(dataset2)
    elif method == "sparse_double":
        transformer = SparsePCA(n_components=10, random_state=0)
        transformer.fit(dataset1)
        pca_data1 = transformer.transform(dataset1)
        pca_data2 = transformer.transform(dataset2)
    elif method == "truncated_double":
        svd = TruncatedSVD(n_components=10, n_iter=7, random_state=42)
        svd.fit(dataset1)
        pca_data1 = svd.transform(dataset1)
        pca_data2 = svd.transform(dataset2)
    elif method == "kernel_double":
        transformer = KernelPCA(n_components=10, eigen_solver='arpack')
        transformer.fit(dataset1)
        pca_data1 = transformer.transform(dataset1)
        pca_data2 = transformer.transform(dataset2)
    else:
        raise RuntimeError("Must enter a valid PCA method")

    return pca_data1, pca_data2


def LPP(dataset, dataset2=None, n_components=10):
    """
    Research extension 3. Locality Preserving Projections.
    """
    # with help from https://github.com/jakevdp/lpproj
    dataset = np.array(dataset)

    # Constructing the adjacency graph and choosing the weights
    # see https://scikit-learn.org/stable/modules/neighbors.html
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors.fit(dataset)
    # see https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html
    weights = kneighbors_graph(neighbors, 5, mode='distance', include_self=True)
    weights = weights.toarray()
    weights = np.maximum(weights, weights.T)

    # diagonal matrix whose entries are column sums of W
    D = np.diag(weights.sum(1))
    L = D - weights  # the Laplacian matrix

    kwargs = dict(eigvals=(0, n_components - 1))
    # eigen decomposition
    eigen = dict()
    # right-hand side of the generalized eigenvector problem
    S, U = linalg.eigh(np.dot(dataset.T, np.dot(D, dataset)), **eigen)
    S[S <= 0] = np.inf
    S_inverse = 1. / np.sqrt(S)
    # left-hand side of the generalized eigenvector problem
    W = S_inverse[:, None] * np.dot(U.T, np.dot(np.dot(dataset.T, np.dot(L, dataset)), U)) * S_inverse

    # finding and applying the projection
    _, projection = linalg.eigh(W, **kwargs)

    if dataset2 is not None:
        return np.dot(dataset, projection), np.dot(dataset2, projection)
    else:
        return np.dot(dataset, projection)
