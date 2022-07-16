import numpy as np
from scipy.spatial import KDTree


def nearest_neighbors(reference, query, k=1, self_query=False, return_distance=False):
    """
    Gets the k nearest neighbors of reference set for each row of the query set

    Parameters
        ----------
        reference : array_like
            An array of points to where nearest neighbors are pulled.
        query : array_like
            An array of points to query nearest neighbors for
        k : int > 0, optional
            the number of nearest neighbors to return
        self_query : bool, optional
            if reference and query are same set of points, set to True
            to avoid each query returning itself as its own nearest neighbor
        return_distance : bool, optional
            if True, return distances of nearest neighbors
    Returns
        -------
        i : integer or array of integers
            The index of each neighbor in reference
            has shape [q, k] where q is number of rows in query
        d : float or array of floats, optional
            if return_distance set to true, returns associated
            euclidean distance of each nearest neighbor
            d is element matched to i, ei the distance of i[a,b] is d[a,b]
            The distances to the nearest neighbors.
    """

    # use scipy's kdtree for extra speed qsar go brrrr
    tree = KDTree(reference)

    if self_query:
        k = [x+2 for x in range(k)]
    else:
        k = [x+1 for x in range(k)]

    d, i = tree.query(query, k=k, workers=-1)

    if return_distance:
        return i, d
    else:
        return i


def modi(data, labels):
    # get all the classes present in the dataset
    classes = np.unique(labels)
    k = classes.shape[0]

    # get the labels of the nearest neighbors
    nn_idx = nearest_neighbors(data, data, k=1, self_query=True)
    nn_labels = labels[nn_idx]

    # calculate the modi
    modi_value = 0
    for c in classes:
        c_arr = np.where(labels == c)[0]
        c_labels = labels[c_arr]
        c_nn_labels = nn_labels[c_arr]

        modi_value += np.sum(c_labels == c_nn_labels) / c_arr.shape[0]

    return (k ** -1) * modi_value


def generate_citations(json_data):
    raise NotImplemented
    # TODO takes in the json parameters of the files and create a citation list that show all the citations a user
    #  should make based on the software and methods used in that dataset
