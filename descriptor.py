import numpy as np
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.DataStructs import ConvertToNumpyArray


class DescriptorCalculator:
    def __init__(self, cache=False):
        self._cache = cache
        self.morgan = None
        self.maccs = None
        self.rdkit = None
        self.mordred = None

    def morgan(self, df, radius=3, n_bits=2048, count=False, use_chirality=False, use_cached=False):
        if self.morgan is not None and use_cached:
            if (radius, n_bits, count, use_chirality) in self.morgan.keys():
                return self.morgan[(radius, n_bits, count, use_chirality)]

        fp = np.zeros(n_bits, dtype=np.int32)

        if count:
            _fp = df["ROMol"].apply(AllChem.GetHashedMorganFingerprint,
                                    kwargs={"radius": radius, "nBits": n_bits, "useChirality": use_chirality})
        else:
            _fp = df["ROMol"].apply(AllChem.GetMorganFingerprintAsBitVect,
                                    kwargs={"radius": radius, "nBits": n_bits, "useChirality": use_chirality})

        ConvertToNumpyArray(_fp, fp)

        if self._cache:
            if self.morgan is None:
                self.morgan = {}
            self.morgan[(radius, n_bits, count, use_chirality)] = fp

        return fp

    def maccs(self, df, use_cached=False):
        if self.maccs is not None and use_cached:
            return self.maccs

        fp = np.zeros(167, dtype=np.int32)

        _fp = df["ROMol"].apply(AllChem.GetMACCSKeysFingerprint)

        ConvertToNumpyArray(_fp, fp)

        if self._cache:
            self.maccs = fp

        return fp

    def rdkit_desc(self, df, return_names=False, use_cached=False):
        if self.rdkit is not None and use_cached:
            if return_names:
                return self.rdkit[0], self.rdkit[1]
            else:
                return self.rdkit[1]

        desc_names = [x[0] for x in Descriptors.descList]
        desc_funcs = [x[1] for x in Descriptors.descList]

        rdkit_desc = np.array(df["ROMol"].apply(lambda x: [func(x) for func in desc_funcs]).to_list())

        if self.cache:
            self.rdkit = (desc_names, rdkit_desc)

        if return_names:
            return desc_names, rdkit_desc
        else:
            return rdkit_desc

    def mordred(self, df, mudra=False, use_cached=False):
        try:
            from mordred import descriptors as mordred_descriptors
            from mordred import Calculator as MordredCalc
        except ImportError:
            raise ImportError("in order to use mordred descriptors you must install the mordred python package")
        if self.mordred is not None and use_cached:
            if mudra in self.mordred.keys():
                return self.mordred[mudra]

        if mudra:
            descriptors = (mordred_descriptors.all[x] for x in range(len(mordred_descriptors.all))
                           if x not in [6, 7, 8, 24, 31, 39, 40, 41])
        else:
            descriptors = mordred_descriptors.all

        calc = MordredCalc(descriptors)

        mordred_desc = np.array([list(x.values()) for x in df["ROMol"].apply(calc)])

        if self._cache:
            if self.mordred is None:
                self.mordred = {}
            self.mordred[mudra] = mordred_desc

        return mordred_desc

    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, value):
        self._cache = value

    @cache.deleter
    def cache(self):
        self.morgan = None
        self.maccs = None
        self.rdkit = None
        self.mordred = None
