import numpy as np
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors as RDKitDescriptors
from rdkit.DataStructs import ConvertToNumpyArray

# TODO ADD SUPPORT FOR NONE PANDAS OBJECTS

# TODO add logging for when descriptors are None, dont want to remove any rows here just return Nan/Nones

# class Descriptor:
#
#     def to_dict(self):
#         raise NotImplementedError
#
#     def get_descriptors(self, romols):
#         raise NotImplementedError
#
#     def to_string(self):
#         raise NotImplementedError
#
#
# class MorganDescriptor(Descriptor):
#     def __init__(self, radius=3, n_bits=2048, count=False, use_chirality=False, use_cached=False):
#         self.radius = radius
#         self.n_bits = n_bits
#         self.count = False
#         self.use_chirality = use_chirality
#         self.use_cached = use_cached
#
#         self.name = "Morgan"
#
#         self.stored_args = {}
#         self.stored_args["radius"] = radius
#         self.stored_args["n_bits"] = n_bits
#         self.stored_args["count"] = count
#         self.stored_args["use_chirality"] = use_chirality
#         self.stored_args["use_cached"] = use_cached
#
#     def to_dict(self):
#
#         d = {}
#         d["Name"] = "MorganDescriptor"
#         d.update(self.stored_args)
#         return d
#
#     def get_descriptors(self, romols):
#
#         # COME BACK AND VECTORIZE
#         if not self.count:
#             _fp = [AllChem.GetHashedMorganFingerprint(x,
#                                                       radius=self.radius,
#                                                       nBits=self.n_bits,
#                                                       useChirality=self.use_chirality) for x in romols]
#         else:
#             raise NotImplementedError
#             _fp = df["ROMol"].apply(AllChem.GetMorganFingerprintAsBitVect,
#                                     kwargs={"radius": radius, "nBits": n_bits, "useChirality": use_chirality})
#
#         fp = []
#         for x in _fp:
#             dest = np.zeros(len(romols), dtype=np.int32)
#             ConvertToNumpyArray(x, dest)
#             fp.append(dest)
#
#         fp = np.vstack(fp)
#
#         return fp
#
#     def get_string(self):
#
#         return f"morgan_fingerprint_radius_{self.radius}_nbits_{self.n_bits}_count_{self.count}_chiral_{self.use_chirality}"


class DescriptorCalculator:
    def __init__(self, cache=False):
        self._cache = cache

    def calc_morgan(self, df, radius=3, n_bits=2048, count=False, use_chirality=False, use_cached=False):
        if hasattr(self, "morgan") and use_cached:
            return self.__getattribute__("morgan")

        if count:
            _fp = df["ROMol"].apply(AllChem.GetHashedMorganFingerprint,
                                    kwargs={"radius": radius, "nBits": n_bits, "useChirality": use_chirality})
        else:
            _fp = df["ROMol"].apply(AllChem.GetMorganFingerprintAsBitVect,
                                    kwargs={"radius": radius, "nBits": n_bits, "useChirality": use_chirality})

        # icky but is there a better way to use the ConvertToNumpy array?
        fp = []
        for x in _fp:
            dest = np.zeros(len(df), dtype=np.int32)
            ConvertToNumpyArray(x, dest)
            fp.append(dest)

        fp = np.vstack(fp)

        if self._cache:
            self.__setattr__("morgan",
                             ({"radius": radius, "n_bits": n_bits, "count": count, "use_chirality": use_chirality}, fp))

        return fp

    def calc_maccs(self, df, use_cached=False):
        if hasattr(self, "maccs") and use_cached:
            return self.maccs[1]

        _fp = df["ROMol"].apply(AllChem.GetMACCSKeysFingerprint)

        # icky but is there a better way to use the ConvertToNumpy array?
        fp = []
        for x in _fp:
            dest = np.zeros(len(df), dtype=np.int32)
            ConvertToNumpyArray(x, dest)
            fp.append(dest)

        fp = np.vstack(fp)

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

        desc_names = [x[0] for x in RDKitDescriptors.descList]
        desc_funcs = [x[1] for x in RDKitDescriptors.descList]

        rdkit_desc = np.array(df["ROMol"].apply(lambda x: [func(x) for func in desc_funcs]).to_list())

        if self.cache:
            self.__setattr__("rdkit", ({"rdkit_desc_names": desc_names}, rdkit_desc))
        else:
            if return_names:
                return desc_names, rdkit_desc
            else:
                return rdkit_desc

    def calc_mordred(self, df, no_rdkit=False, use_cached=False):
        try:
            from mordred import descriptors as mordred_descriptors
            from mordred import Calculator as MordredCalc
        except ImportError:
            raise ImportError("in order to use mordred descriptors you must install the mordred python package")

        if hasattr(self, "mordred") and use_cached:
            return self.mordred

        if no_rdkit:
            descriptors = (mordred_descriptors.all[x] for x in range(len(mordred_descriptors.all))
                           if x not in [6, 7, 8, 24, 31, 39, 40, 41])
        else:
            descriptors = mordred_descriptors.all

        calc = MordredCalc(descriptors)

        mordred_desc = np.array([list(x.values()) for x in df["ROMol"].apply(calc)])

        if self._cache:
            self.__setattr__("mordred", ({"no_rdkit": no_rdkit}, mordred_desc))
        else:
            return mordred_desc

    def calc_custom_descriptor(self, df, name, func, **kwargs):
        desc = func(df, **kwargs)
        if self._cache:
            self.__setattr__(f"calc_{name}", func)
            self.__setattr__(name, (kwargs, desc))
        else:
            return desc

    def add_descriptor(self, name, desc, **kwargs):
        if name in self.__dict__:
            raise ValueError("descriptor already exists to overwrite use set_descriptor")
        self.set_descriptor(name, desc, **kwargs)

    def set_descriptor(self, name, desc, **kwargs):
        if self._cache:
            self.__setattr__(name, (kwargs, desc))
        else:
            return desc

    def get_available_descriptors(self):
        return [key for key in self.__dict__.keys() if key != "_cache"]

    def get_descriptor(self, name):
        if name not in self.get_available_descriptors():
            raise ValueError(f"descriptor {name} does not exists for given dataset")
        return self.__getattribute__(name)

    def delete_descriptor(self, name):
        if name in self.__dict__:
            self.__delattr__(name)

    def get_all_descriptors(self, return_settings=False):
        if return_settings:
            return [(key, val[0], val[1]) for key, val in self.__dict__.items() if key != "_cache"]
        else:
            return [(key, val[1]) for key, val in self.__dict__.items() if key != "_cache"]

    def func_exists(self, name):
        func_call = "calc_" + str(name)
        return func_call in dir(self) and callable(self.__getattribute__(func_call))

    def get_descriptor_funcs(self):
        return [x.replace("calc_", "") for x in dir(self) if callable(self.__getattribute__(x) and "calc_" in x)]

    def iter_descriptors(self, return_settings=False):
        if self._cache:
            for key, val in self.__dict__:
                if key != "_cache":
                    if return_settings:
                        yield key, val[0], val[1]
                    else:
                        yield key, val[1]

    def to_dict(self, name):
        args, _ = self.get_descriptor(name)
        args["name"] = name
        return args

    def get_string(self, name):
        args, _ = self.get_descriptor(name)
        return "_".join(name + list(args.values()))

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

