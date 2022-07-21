import numpy as np
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.DataStructs import ConvertToNumpyArray

# TODO ADD SUPPORT FOR NONE PANDAS OBJECTS

# TODO add logging for when descriptors are None, dont want to remove any rows here just return Nan/Nones

class MorganDescriptor:

    def __init__(self, radius=3, n_bits=2048, count=False, use_chirality=False, use_cached=False):
        self.radius = radius
        self.n_bits = n_bits
        self.count = False
        self.use_chirality = use_chirality
        self.use_cached = use_cached

        self.name = "Morgan"

        self.stored_args = {}
        self.stored_args["radius"] = radius
        self.stored_args["n_bits"] = n_bits
        self.stored_args["count"] = count
        self.stored_args["use_chirality"] = use_chirality
        self.stored_args["use_cached"] = use_cached

    def to_dict(self):

        d = {}
        d["Name"] = "MorganDecriptor"
        d.update(self.stored_args)
        return d

    def get_descriptors(self, romols):

        #COME BACK AND VECTORIZE
        if not self.count:
            _fp = [AllChem.GetHashedMorganFingerprint(x,
                                    radius=self.radius, 
                                    nBits = self.n_bits,
                                    useChirality = self.use_chirality) for x in romols]
        else:
            raise NotImplementedError
            _fp = df["ROMol"].apply(AllChem.GetMorganFingerprintAsBitVect,
                                    kwargs={"radius": radius, "nBits": n_bits, "useChirality": use_chirality})

        fp = []
        for x in _fp:
            dest = np.zeros(len(romols), dtype=np.int32)
            ConvertToNumpyArray(x, dest)
            fp.append(dest)

        fp = np.vstack(fp)

        return fp




class DescriptorCalculator:
    def __init__(self, cache=False):
        self._cache = cache

    def calc_morgan(self, df, radius=3, n_bits=2048, count=False, use_chirality=False, use_cached=False):
        if hasattr(self, "morgan") and use_cached:
            return self.__getattribute__("morgan")

        fp = np.zeros(n_bits, dtype=np.int32)

        #COME BACK AND VECTORIZE
        if count:
            _fp = [AllChem.GetHashedMorganFingerprint(x,
                                    radius=radius, 
                                    nBits = n_bits,
                                    useChirality = use_chirality) for x in df["ROMol"]]
        else:
            _fp = df["ROMol"].apply(AllChem.GetMorganFingerprintAsBitVect,
                                    kwargs={"radius": radius, "nBits": n_bits, "useChirality": use_chirality})

        fp = []
        for x in _fp:
            dest = np.zeros(len(df), dtype=np.int32)
            ConvertToNumpyArray(x, dest)
            fp.append(dest)

        fp = np.vstack(fp)

        if self._cache:
            self.__setattr__("morgan", ({"radius": radius, "n_bits": n_bits, "count": count, "use_chirality": use_chirality}, fp))

        return fp

    def calc_maccs(self, df, use_cached=False):
        if hasattr(self, "maccs") and use_cached:
            return self.maccs

        fp = np.zeros(167, dtype=np.int32)

        _fp = df["ROMol"].apply(AllChem.GetMACCSKeysFingerprint)

        ConvertToNumpyArray(_fp, fp)

        if self._cache:
            self.__setattr__("maccs", ({}, fp))
        else:
            return fp

    def calc_rdkit(self, df, return_names=False, use_cached=False):
        if hasattr(self, "rdkit") and use_cached:
            if return_names:
                return self.rdkit[0], self.rdkit[1]
            else:
                return self.rdkit[1]

        desc_names = [x[0] for x in Descriptors.descList]
        desc_funcs = [x[1] for x in Descriptors.descList]

        rdkit_desc = np.array(df["ROMol"].apply(lambda x: [func(x) for func in desc_funcs]).to_list())

        if self.cache:
            self.__setattr__("rdkit", ({"rdkit_desc_names": desc_names}, rdkit_desc))
        else:
            if return_names:
                return desc_names, rdkit_desc
            else:
                return rdkit_desc

    def calc_mordred(self, df, mudra=False, use_cached=False):
        try:
            from mordred import descriptors as mordred_descriptors
            from mordred import Calculator as MordredCalc
        except ImportError:
            raise ImportError("in order to use mordred descriptors you must install the mordred python package")
        if hasattr(self, "mordred") and use_cached:
            return self.mordred

        if mudra:
            descriptors = (mordred_descriptors.all[x] for x in range(len(mordred_descriptors.all))
                           if x not in [6, 7, 8, 24, 31, 39, 40, 41])
        else:
            descriptors = mordred_descriptors.all

        calc = MordredCalc(descriptors)

        mordred_desc = np.array([list(x.values()) for x in df["ROMol"].apply(calc)])

        if self._cache:
            self.__setattr__("mordred", ({}, mordred_desc))
        else:
            return mordred_desc
    
    def calc_custom(self, df, name, func, kwargs=None):
        # TODO add in support to save custom descriptor calculations to the calc object so that the dataset object can
        #  auto detect them
        desc = func(df, **kwargs)

        if self._cache:
            self.__setattr__(name, (kwargs, desc))
        else:
            return desc
        
    def add_descriptor(self, name, desc, options=None):
        if self._cache:
            if options is None:
                options = {}
            self.__setattr__(name, (options, desc))

    def get_all_descriptors(self):
        return [(key, val[0], val[1]) for key, val in self.__dict__.items() if key != "_cache"]

    def func_exists(self, name):
        func_call = "calc_" + str(name)
        return func_call in dir(self) and callable(self.__getattribute__(func_call))

    def get_descriptor_funcs(self):
        return [x.replace("calc_", "") for x in dir(self) if callable(self.__getattribute__(x) and "calc_" in x)]

    def iter_descriptors(self):
        if self._cache:
            for key, val in self.__dict__:
                if key != "_cache":
                    yield key, val[0], val[1]

    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, value):
        self._cache = value

    @cache.deleter
    def cache(self):
        for attr in self.__dict__.keys():
            if attr != "_cache":
                self.__setattr__(attr, None)
