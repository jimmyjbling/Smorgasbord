import numpy as np
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors as RDKitDescriptors
from rdkit.DataStructs import ConvertToNumpyArray

# TODO ADD SUPPORT FOR NONE PANDAS OBJECTS

# TODO add logging for when descriptors are None, dont want to remove any rows here just return Nan/Nones


class DescriptorCalculator:
    # Note all descriptor methods need to be able to handle failures by returning numpy nans for that row
    def __init__(self):
        pass

    @staticmethod
    def calc_morgan(df, radius=3, n_bits=2048, count=False, use_chirality=False):
        if count:
            _fp = df["ROMol"].apply(lambda x: AllChem.GetHashedMorganFingerprint(x, radius=radius, nBits=n_bits, useChirality=use_chirality) if x is not None else np.nan)
        else:
            _fp = df["ROMol"].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, radius=radius, nBits=n_bits, useChirality=use_chirality) if x is not None else np.nan)

        # icky but is there a better way to use the ConvertToNumpy array?
        fp = []
        for x in _fp:
            if isinstance(x, float) and np.isnan(x):
                fp.append(np.full(n_bits, np.nan))
            else:
                dest = np.zeros(n_bits, dtype=np.int32)
                ConvertToNumpyArray(x, dest)
                fp.append(dest)
        fp = np.vstack(fp).astype(float)

        return fp

    @staticmethod
    def calc_maccs(df):
        _fp = df["ROMol"].apply(lambda x: AllChem.GetMACCSKeysFingerprint(x) if x is not None else np.nan)

        # icky but is there a better way to use the ConvertToNumpy array?
        fp = []
        for x in _fp:
            if isinstance(x, float) and np.isnan(x):
                fp.append(np.full(167, np.nan))
            else:
                dest = np.zeros(167, dtype=np.int32)
                ConvertToNumpyArray(x, dest)
                fp.append(dest)
        fp = np.vstack(fp).astype(float)

        return fp

    @staticmethod
    def calc_rdkit(df):
        desc_funcs = [x[1] for x in RDKitDescriptors.descList]

        rdkit_desc = np.array(df["ROMol"].apply(lambda x: [func(x) for func in desc_funcs] if x is not None else [np.nan for _ in range(len(desc_funcs))]).to_list()).astype(float)

        return rdkit_desc

    @staticmethod
    def calc_mordred(df, no_rdkit=False):
        try:
            from mordred import descriptors as mordred_descriptors
            from mordred import Calculator as MordredCalc
        except ImportError:
            raise ImportError("in order to use mordred descriptors you must install the mordred python package")

        if no_rdkit:
            descriptors = (mordred_descriptors.all[x] for x in range(len(mordred_descriptors.all))
                           if x not in [6, 7, 8, 24, 31, 39, 40, 41])
        else:
            descriptors = mordred_descriptors.all

        calc = MordredCalc(descriptors)

        # when list comprehension goes wild
        mordred_desc = np.array([list(x.values()) if x is not None else [np.nan for _ in range(len(calc))] for x in
                                 df["ROMol"].apply(lambda z: calc(z) if z is not None else None)]).astype(float)

        return mordred_desc

    @staticmethod
    def rdkit_desc_name():
        return [x[0] for x in RDKitDescriptors.descList]

    @ staticmethod
    def _get_func_name(name):
        if not name.startswith("calc_"):
            name = f"calc_{name}"
        return name

    def set_descriptor_function(self, name, func):
        self.__setattr__(self._get_func_name(name), func)

    def _get_available_funcs(self):
        return [x for x in dir(self) if x.startswith("calc_") and callable(self.__getattribute__(x))]

    def func_exists(self, name):
        func_call = self._get_func_name(name)
        return func_call in dir(self) and callable(self.__getattribute__(func_call))

    def get_descriptor_func(self, name):
        if self.func_exists(name):
            return self.__getattribute__(self._get_func_name(name))
        else:
            raise ValueError(f"descriptor {name} does not exist")

    def funcs(self):
        for x in self._get_available_funcs():
            yield self.__getattribute__(x)


class DatasetDescriptorCalculator(DescriptorCalculator):
    def __init__(self, cache=True):
        super().__init__()
        self._cache = cache

    def add_calculated_descriptor(self, name, desc, **kwargs):
        if name in self.__dict__:
            raise ValueError("descriptor already exists to overwrite use set_descriptor")
        self.set_calculated_descriptor(name, desc, **kwargs)

    def set_calculated_descriptor(self, name, desc, **kwargs):
        if self._cache:
            self.__setattr__(name, (kwargs, desc))
        else:
            return desc

    def delete_calculated_descriptor(self, name):
        if name in self.__dict__:
            self.__delattr__(name)

    def _get_available_calculated(self):
        return [x for x in dir(self) if x != "_cache" and not callable(self.__getattribute__(x))]

    def calculated_exists(self, name):
        return name in self._get_available_calculated()

    def _calculate_descriptor(self, name, df, **kwargs):
        desc = self.get_descriptor_func(name)(df, **kwargs)
        if self._cache:
            self.__setattr__(name, (kwargs, self.get_descriptor_func(name)(df, **kwargs)))
        return desc

    # When try to get descriptors from the dataset object, you should always interface with this function to get them
    def get_descriptor_value(self, name, df, **kwargs):
        if not self.calculated_exists(name) or not self._args_match(name, kwargs):
            return self._calculate_descriptor(name, df, **kwargs)
        else:
            return self.__getattribute__(name)[1]

    def get_descriptor_args(self, name):
        if not self.calculated_exists(name):
            raise ValueError(f"descriptor {name} has not been calculated for the given dataset")
        return self.__getattribute__(name)[0]

    def _args_match(self, name, args):
        return all([args[x] == self.get_descriptor_args(name)[x] for x in args.keys()])

    def values(self):
        for x in self._get_available_calculated():
            yield self.__getattribute__(x)[1]

    def args(self):
        for x in self._get_available_calculated():
            yield self.__getattribute__(x)[0]

    def args_and_vals(self):
        for x in self._get_available_calculated():
            yield self.__getattribute__(x)

    def to_dict(self, name):
        args = self.get_descriptor_args(name)
        args["name"] = name
        return args

    def get_string(self, name):
        args = self.get_descriptor_args(name)
        return "_".join([name] + [f'{key}:{val}' for key, val in args.items()])

    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, value):
        self._cache = bool(value)

    @cache.deleter
    def cache(self):
        for attr in self.__dict__.keys():
            if attr != "_cache":
                self.__setattr__(attr, None)

