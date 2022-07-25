import numpy as np
from scipy import spatial as sp


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
    tree = sp.KDTree(reference)

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
        c_nn_labels = nn_labels[c_arr].transpose()
        modi_value += np.sum(c_labels == c_nn_labels) / c_arr.shape[0]

    return (k ** -1) * modi_value


def generate_citations(json_data):
    raise NotImplemented
    # TODO takes in the json parameters of the files and create a citation list that show all the citations a user
    #  should make based on the software and methods used in that dataset


def apd_screening(y, training_data=None, threshold=None, norm_func=None):
    """
    screen the AD of a set of datapoint using the method outlined in [1]

    [1]: S. Zhang, A. Golbraikh, S. Oloff, H. Kohn, A. Tropsha J. Chem. Inf. Model., 46 (2006), pp. 1984–1995

    Parameters
    ----------
        y : array-like
            data to screen for AD
        training_data : array-like, optional
            dataset that defines the AD. Not required if threshold is not None
            will override any passed threshold and norm_func
        threshold: float, optional
            the threshold to make AD cutoff decisions. Must be non-None if training_data is None
        norm_func: func, optional
            if training_data is None, will define the norm func to use on y.
            **warning** if training_data is None, y will not be normalized unless this function is passed.
                        this can induce major errors if y was not properly normalized ahead of time. it is
                        recommended that this is passed is training_data is None.
    threshold
    norm_func

    Returns
    -------

    """
    if training_data is None and threshold is None:
        raise ValueError("training_data or threshold must be set both were None")

    if training_data is not None:
        training_data, norm_func = normalize(training_data, return_normalize_function=True)
        threshold = apd_threshold(training_data)

    if norm_func is not None:
        y = norm_func(y)

    _, dist = nearest_neighbors(training_data, y, k=1, self_query=False, return_distance=True)

    return np.where(np.concatenate(dist) < threshold, 1, 0)


def apd_threshold(X, z=0.5, normalize_data=False):
    """
    Gets the applicability domain threshold from [1] with euclidean distance
    AD = d` + zs`
    where d` is the mean(d) where d is every pairwise distance less than the mean of all pairwise distance and s`
    follows similar logic for standard deviation

    [1]: S. Zhang, A. Golbraikh, S. Oloff, H. Kohn, A. Tropsha J. Chem. Inf. Model., 46 (2006), pp. 1984–1995

    Parameters
    ----------
        X : array-like
            An array of the datapoint to calculate AD threshold on.
        z : float, optional default=0.5
            The scalar that operate on the std of data-points below the mean
        normalize_data : bool, optional default=False
            if true will normalize the data (but will not save the normalization function **not recommended**
            you should normalize before by calling normalize so you can save the normalization function.
    Returns
    -------
        threshold : float
            threshold for AD screening
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    if normalize_data:
        X = normalize(X)

    d = sp.distance.pdist(X)
    d_prime = d[np.where(d < np.mean(d))]

    m_prime = np.mean(d_prime)
    sd_prime = np.std(d_prime)
    return m_prime + (z * sd_prime)


def normalize(x, return_normalize_function=False):
    """
    Normalize a 2d array of numbers to be between 0-1 based on the max value of each column of the array,

    Parameters
    ----------
        x : array-like
            1d array of numerics to normalize
        return_normalize_function : bool, optional default = False
            return the max value for the column

    Returns
    -------
        transformed array of the same shape of x with the normalized values
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    res = np.apply_along_axis(lambda a: a / a.max(), 0, x)

    if return_normalize_function:
        col_max = np.apply_along_axis(np.max, 0, x)
        return res, lambda a: a / col_max
    else:
        return res
