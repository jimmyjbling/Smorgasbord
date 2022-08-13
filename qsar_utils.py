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
        k = [x + 2 for x in range(k)]
    else:
        k = [x + 1 for x in range(k)]

    d, i = tree.query(query, k=k, workers=-1)

    if return_distance:
        return i, d
    else:
        return i


def modi(data, labels, return_class_contribution=False):
    """
        Gets the MODI from the given data and label set
        Parameters
        ----------
            data : array_like
                An array chemical descriptors (rows are chemicals and columns are descriptors).
            labels : array_like
                An array labels that are row matched to the data array
            return_class_contribution : bool, optional
                if True, return the normalized MODI for each class. Useful for imbalanced datasets
        Returns
        -------
            modi : float
                the calculated MODI for the given data and label
            class_contrib : list of tuples of length 2 (str, float), optional
                if return_class_contribution set to true, returns associated
                MODI score for each class in the data as a tuple of (class, MODI)
    """

    # get all the classes present in the dataset
    classes = np.unique(labels)
    k = classes.shape[0]

    # get the labels of the nearest neighbors
    nn_idx = nearest_neighbors(data, data, k=1, self_query=True)
    nn_labels = labels[nn_idx]

    # calculate the modi
    modi_value = 0
    class_contrib = []
    for c in classes:
        c_arr = np.where(labels == c)[0]
        c_labels = labels[c_arr]
        c_nn_labels = nn_labels[c_arr].flatten()
        modi_value += np.sum(c_labels == c_nn_labels) / c_arr.shape[0]
        class_contrib.append(np.sum(c_labels == c_nn_labels) / c_arr.shape[0])

    if not return_class_contribution:
        return (k ** -1) * modi_value
    else:
        return (k ** -1) * modi_value, class_contrib


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


def get_morgan_finger(mol, nbits=2048, radius=3, chiral=False, count=False, bit_info=False):
    """
    Get count morgan fingerprint from RDKit Mol object
    """
    from rdkit.Chem import AllChem
    from rdkit.DataStructs import ConvertToNumpyArray

    fp = np.ones(nbits, dtype=int)
    bi = {}
    if count:
        ConvertToNumpyArray(AllChem.GetHashedMorganFingerprint(mol, nBits=nbits, radius=radius,
                                                               useChirality=chiral, bitInfo=bi), fp)
    else:
        ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, nBits=nbits, radius=radius,
                                                                  useChirality=chiral, bitInfo=bi), fp)

    if bit_info:
        return fp, bi
    else:
        return fp


def chemical_diversity(mols, nbits=2048, radius=3, chiral=False):
    """
    Calculate a measure of how diverse a given set of molecules is

    Uses morgan count fingerprints to convert chemicals to a coordinates and returns the mean and std of the pairwise
    distances of all combinations. This can be compared to the values for chembl for the same morgan settings to
    get an idea of how diverse the chemicals are. For reference with 2048 bit the max distance is srqt(2048) or 45.2548

    Parameters
    ----------
        mols: pandas series of rdkit Mol objects
            The molecules you want to calculate diversity for
        nbits: int, optional default=2048
            The number of bits for the fingerprint
        radius: int, optional default=3
            The radius of the fingerprint
        chiral: bool, optional default=False
            Use chiral fingerprints
    Returns
    -------
        diversity_score: tuple
            returns a tuple of (mean_pairwise, std_pairwise)
    """
    fps = np.vstack(mols[mols.columns[0]].apply(get_morgan_finger, nbits=nbits, radius=radius,
                                                chiral=chiral, count=True).to_numpy())
    dist = sp.distance.pdist(fps)

    return np.mean(dist), np.std(dist)


def umap(X, labels, save_loc=None, **kwargs):
    """
    Calculate and plots the UMap for a given set of chemical descriptors

    Parameters
    ----------
        X: array-like
            array of chemical descriptors
        labels: array-like
            array of shape [n, 1] of classification labels for umap
        save_loc: str, optional
            if valid file location is passed will save umap plot to file rather than showing plot
        kwargs: optional
            kwargs to be passed to the umap function

    Returns
    -------
    None

    """
    import umap as _umap
    import matplotlib.pyplot as plt

    mapper = _umap.UMAP(**kwargs).fit(X)
    _umap.plot.points(mapper, labels=labels)

    if save_loc is not None:
        plt.savefig(save_loc)
    else:
        plt.show()


def get_finger_bit_substructure(mol, bit, nbits=2048, radius=3, chiral=False):
    # TODO I think this will artificially close rings that are not known to be closed
    # example 'Cc1c(C(O)CN2CCC3(CC2)CC(=O)N(c2ccc(S(C)(=O)=O)cn2)C3)ccc2c1COC2=O' bits 1024 radius 3 chiral with
    #  bit 889 this 5 member ring should not be closed but the SMARTS has it closed

    """
    Will find the substructure of a given molecule at a given morgan bit and return it as a SMARTS for matching
    plotting of the shape and a rdkit mol

    Parameters
    ----------
        mol: rdkit Mol object
            molecule that we want to collect bit substructures for
        bit: int
            the morgan bit you want to get the substructure of
        nbits: int, optional default=2048
            The number of bits for the fingerprint
        radius: int, optional default=3
            The radius of the fingerprint
        chiral: bool, optional default=False
            Use chiral fingerprints

    Returns
    -------
        if bit is not set in mol returns None
        else returns list of SMARTS for each sub structure in that bit and List of substructure mol objects

    """
    from rdkit.Chem import AllChem
    from rdkit import Chem

    _fp, bi = AllChem.GetMorganFingerprintAsBitVect(mol, nBits=nbits, radius=radius, useChirality=chiral, bitInfo=True)

    if bit not in bi.keys():
        return None

    sub_mol_smarts = []
    sub_mols = []

    for (idx, radius) in bi[bit]:
        # get all the bonds as a path in the given environment
        bit_path = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, idx)

        # get all the atom ids that attributed to the environment
        # this will miss the last radius of bonds
        atom_idx = {idx}
        for bp in bit_path:
            atom_idx.add(mol.GetBondWithIdx(bp).GetBeginAtomIdx())
            atom_idx.add(mol.GetBondWithIdx(bp).GetEndAtomIdx())

        # collect the bonds and dummy atoms leaving the edge atoms of the bit to make into smarts
        enlarged_env = set()
        for atom in atom_idx:
            a = mol.GetAtomWithIdx(atom)
            for b in a.GetBonds():
                b_idx = b.GetIdx()
                if b_idx not in bit_path:
                    enlarged_env.add(b_idx)
        enlarged_env = list(enlarged_env)
        enlarged_env += bit_path

        # covert into sub mol and save index map
        atom_map = {}
        sub_mol = Chem.PathToSubmol(mol, enlarged_env, atomMap=atom_map)

        # covert the atoms to dummy atoms if they were added to only get their bonds
        for map_idx in atom_map.keys():
            if map_idx not in atom_idx:
                sub_mol.GetAtomWithIdx(atom_map[map_idx]).SetAtomicNum(0)
                sub_mol.GetAtomWithIdx(atom_map[map_idx]).UpdatePropertyCache()

        sub_mol_smarts.append(Chem.MolToSmarts(sub_mol))
        sub_mols.append(sub_mol)

    return sub_mol_smarts, sub_mols


def smote(X, y):
    """
    Implement the SMOTE method to balance your dataset. This method of balancing is not compatible with plates

    SMOTE and variations of it will create new artificial data in an attempt to augment a dataset. This data
    will then lack a real chemical structure that associated with it (creates new descriptors not new smiles).
    The entire point of Smorgasbord and the plate approach is that the data has known chemicals behind them.
    Since SMOTE violates this it cannot be used inside the vanilla plate workflow and can only be used in manual
    workflow processes. It is unadvised to use it for chemical data for that aforementioned reasons

    Parameters
    ----------
        X: array-like 2D
            array of data to sample from
        y: array-like 1D
            array of labels for given data to determine what to sample from

    Returns
    ----------
        X: array-like 2D
            array of resampled data
        y: array-like 1D
            array of resembled labels
    """
    from imblearn.over_sampling import SMOTE

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    return X_res, y_res


def LoadSDF(filename, idName='ID', molColName='ROMol', includeFingerprints=False,
            isomericSmiles=True, smilesName=None, embedProps=False, removeHs=True,
            strictParsing=True):
    '''
    Read file in SDF format and return as Pandas data frame.
    If embedProps=True all properties also get embedded in Mol objects in the molecule column.
    If molColName=None molecules would not be present in resulting DataFrame (only properties
    would be read).
    '''

    from rdkit import Chem
    from rdkit.Chem.PandasTools import _MolPlusFingerprint, RenderImagesInAllDataFrames
    import pandas as pd
    from logging import log

    if isinstance(filename, str):
        if filename.lower()[-3:] == ".gz":
            import gzip
            f = gzip.open(filename, "rb")
        else:
            f = open(filename, 'rb')
        close = f.close
    else:
        f = filename
        close = None  # don't close an open file that was passed in
    records = []
    indices = []
    sanitize = bool(molColName is not None or smilesName is not None)
    for i, mol in enumerate(
            Chem.ForwardSDMolSupplier(f, sanitize=sanitize, removeHs=removeHs,
                                      strictParsing=strictParsing)):
        if mol is None:
            print("Shit")
            continue
        row = dict((k, mol.GetProp(k)) for k in mol.GetPropNames())
        if molColName is not None and not embedProps:
            for prop in mol.GetPropNames():
                mol.ClearProp(prop)
        if mol.HasProp('_Name'):
            row[idName] = mol.GetProp('_Name')
        if smilesName is not None:
            try:
                row[smilesName] = Chem.MolToSmiles(mol, isomericSmiles=isomericSmiles)
            except Exception:
                log.warning('No valid smiles could be generated for molecule %s', i)
                row[smilesName] = None
        if molColName is not None and not includeFingerprints:
            row[molColName] = mol
        elif molColName is not None:
            row[molColName] = _MolPlusFingerprint(mol)
        records.append(row)
        indices.append(i)

    if close is not None:
        close()
    RenderImagesInAllDataFrames(images=True)
    return pd.DataFrame(records, index=indices)

LoadSDF(f"C:\\Users\\welln\\OneDrive\\TropshaLab\\Projects\\Smorgasbord\\SDF-FOR-MODI\\DRASTIC\\{1}.sdf")
