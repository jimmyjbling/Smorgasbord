import numpy as np
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem import Descriptors, Crippen, MolSurf, Lipinski, Fragments, EState, GraphDescriptors

# TODO ADD SUPPORT FOR NONE PANDAS OBJECTS

# TODO add logging for when descriptors are None, dont want to remove any rows here just return Nan/Nones

class Descriptor:

    def to_dict(self):
        raise NotImplementedError

    def get_descriptors(self, romols):
        raise NotImplementedError

    def to_string(self):
        raise NotImplementedError

class MorganDescriptor(Descriptor):

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

    def get_string(self):

        return f"morgan_fingerprint_radius_{self.radius}_nbits_{self.n_bits}_count_{self.count}_chiral_{self.use_chirality}"

class RDKitDescriptor(Descriptor):

    def __init__(self):
        self.name = "RDKit"
        pass

    def to_dict(self):
        return {}

    def get_descriptors(self, romols):

        return np.array([self.get_all_rdkit_descriptors(x) for x in romols])

    def get_string(self):

        return "rdkitdesc"


    def get_all_rdkit_descriptors(self, mol):

        descriptor_functions = np.array([Crippen.MolLogP,
                                      Crippen.MolMR,
                                      Descriptors.FpDensityMorgan1,
                                      Descriptors.FpDensityMorgan2,
                                      Descriptors.FpDensityMorgan3,
                                      Descriptors.FractionCSP3,
                                      Descriptors.HeavyAtomMolWt,
                                      Descriptors.MaxAbsPartialCharge,
                                      Descriptors.MaxPartialCharge,
                                      Descriptors.MinAbsPartialCharge,
                                      Descriptors.MinPartialCharge,
                                      Descriptors.MolWt,
                                      Descriptors.NumRadicalElectrons,
                                      Descriptors.NumValenceElectrons,
                                      EState.EState.MaxAbsEStateIndex,
                                      EState.EState.MaxEStateIndex,
                                      EState.EState.MinAbsEStateIndex,
                                      EState.EState.MinEStateIndex,
                                      EState.EState_VSA.EState_VSA1,
                                      EState.EState_VSA.EState_VSA10,
                                      EState.EState_VSA.EState_VSA11,
                                      EState.EState_VSA.EState_VSA2,
                                      EState.EState_VSA.EState_VSA3,
                                      EState.EState_VSA.EState_VSA4,
                                      EState.EState_VSA.EState_VSA5,
                                      EState.EState_VSA.EState_VSA6,
                                      EState.EState_VSA.EState_VSA7,
                                      EState.EState_VSA.EState_VSA8,
                                      EState.EState_VSA.EState_VSA9,
                                      Fragments.fr_Al_COO,
                                      Fragments.fr_Al_OH,
                                      Fragments.fr_Al_OH_noTert,
                                      Fragments.fr_aldehyde,
                                      Fragments.fr_alkyl_carbamate,
                                      Fragments.fr_alkyl_halide,
                                      Fragments.fr_allylic_oxid,
                                      Fragments.fr_amide,
                                      Fragments.fr_amidine,
                                      Fragments.fr_aniline,
                                      Fragments.fr_Ar_COO,
                                      Fragments.fr_Ar_N,
                                      Fragments.fr_Ar_NH,
                                      Fragments.fr_Ar_OH,
                                      Fragments.fr_ArN,
                                      Fragments.fr_aryl_methyl,
                                      Fragments.fr_azide,
                                      Fragments.fr_azo,
                                      Fragments.fr_barbitur,
                                      Fragments.fr_benzene,
                                      Fragments.fr_benzodiazepine,
                                      Fragments.fr_bicyclic,
                                      Fragments.fr_C_O,
                                      Fragments.fr_C_O_noCOO,
                                      Fragments.fr_C_S,
                                      Fragments.fr_COO,
                                      Fragments.fr_COO2,
                                      Fragments.fr_diazo,
                                      Fragments.fr_dihydropyridine,
                                      Fragments.fr_epoxide,
                                      Fragments.fr_ester,
                                      Fragments.fr_ether,
                                      Fragments.fr_furan,
                                      Fragments.fr_guanido,
                                      Fragments.fr_halogen,
                                      Fragments.fr_hdrzine,
                                      Fragments.fr_hdrzone,
                                      Fragments.fr_HOCCN,
                                      Fragments.fr_imidazole,
                                      Fragments.fr_imide,
                                      Fragments.fr_Imine,
                                      Fragments.fr_isocyan,
                                      Fragments.fr_isothiocyan,
                                      Fragments.fr_ketone,
                                      Fragments.fr_ketone_Topliss,
                                      Fragments.fr_lactam,
                                      Fragments.fr_lactone,
                                      Fragments.fr_methoxy,
                                      Fragments.fr_morpholine,
                                      Fragments.fr_N_O,
                                      Fragments.fr_Ndealkylation1,
                                      Fragments.fr_Ndealkylation2,
                                      Fragments.fr_NH0,
                                      Fragments.fr_NH1,
                                      Fragments.fr_NH2,
                                      Fragments.fr_Nhpyrrole,
                                      Fragments.fr_nitrile,
                                      Fragments.fr_nitro,
                                      Fragments.fr_nitro_arom,
                                      Fragments.fr_nitro_arom_nonortho,
                                      Fragments.fr_nitroso,
                                      Fragments.fr_oxazole,
                                      Fragments.fr_oxime,
                                      Fragments.fr_para_hydroxylation,
                                      Fragments.fr_phenol,
                                      Fragments.fr_phenol_noOrthoHbond,
                                      Fragments.fr_phos_acid,
                                      Fragments.fr_phos_ester,
                                      Fragments.fr_piperdine,
                                      Fragments.fr_piperzine,
                                      Fragments.fr_priamide,
                                      Fragments.fr_prisulfonamd,
                                      Fragments.fr_pyridine,
                                      Fragments.fr_quatN,
                                      Fragments.fr_SH,
                                      Fragments.fr_sulfide,
                                      Fragments.fr_sulfonamd,
                                      Fragments.fr_sulfone,
                                      Fragments.fr_term_acetylene,
                                      Fragments.fr_tetrazole,
                                      Fragments.fr_thiazole,
                                      Fragments.fr_thiocyan,
                                      Fragments.fr_thiophene,
                                      Fragments.fr_unbrch_alkane,
                                      Fragments.fr_urea,
                                      GraphDescriptors.BalabanJ,
                                      GraphDescriptors.BertzCT,
                                      GraphDescriptors.Chi0,
                                      GraphDescriptors.Chi0n,
                                      GraphDescriptors.Chi0v,
                                      GraphDescriptors.Chi1,
                                      GraphDescriptors.Chi1n,
                                      GraphDescriptors.Chi1v,
                                      GraphDescriptors.Chi2n,
                                      GraphDescriptors.Chi2v,
                                      GraphDescriptors.Chi3n,
                                      GraphDescriptors.Chi3v,
                                      GraphDescriptors.Chi4n,
                                      GraphDescriptors.Chi4v,
                                      GraphDescriptors.HallKierAlpha,
                                      GraphDescriptors.Ipc,
                                      GraphDescriptors.Kappa1,
                                      GraphDescriptors.Kappa2,
                                      GraphDescriptors.Kappa3,
                                      Lipinski.HeavyAtomCount,
                                      Lipinski.NHOHCount,
                                      Lipinski.NOCount,
                                      Lipinski.NumAliphaticCarbocycles,
                                      Lipinski.NumAliphaticHeterocycles,
                                      Lipinski.NumAliphaticRings,
                                      Lipinski.NumAromaticCarbocycles,
                                      Lipinski.NumAromaticHeterocycles,
                                      Lipinski.NumAromaticRings,
                                      Lipinski.NumHAcceptors,
                                      Lipinski.NumHDonors,
                                      Lipinski.NumHeteroatoms,
                                      Lipinski.NumRotatableBonds,
                                      Lipinski.NumSaturatedCarbocycles,
                                      Lipinski.NumSaturatedHeterocycles,
                                      Lipinski.NumSaturatedRings,
                                      Lipinski.RingCount,
                                      MolSurf.LabuteASA,
                                      MolSurf.PEOE_VSA1,
                                      MolSurf.PEOE_VSA10,
                                      MolSurf.PEOE_VSA11,
                                      MolSurf.PEOE_VSA12,
                                      MolSurf.PEOE_VSA13,
                                      MolSurf.PEOE_VSA14,
                                      MolSurf.PEOE_VSA2,
                                      MolSurf.PEOE_VSA3,
                                      MolSurf.PEOE_VSA4,
                                      MolSurf.PEOE_VSA5,
                                      MolSurf.PEOE_VSA6,
                                      MolSurf.PEOE_VSA7,
                                      MolSurf.PEOE_VSA8,
                                      MolSurf.PEOE_VSA9,
                                      MolSurf.SlogP_VSA1,
                                      MolSurf.SlogP_VSA10,
                                      MolSurf.SlogP_VSA11,
                                      MolSurf.SlogP_VSA12,
                                      MolSurf.SlogP_VSA2,
                                      MolSurf.SlogP_VSA3,
                                      MolSurf.SlogP_VSA4,
                                      MolSurf.SlogP_VSA5,
                                      MolSurf.SlogP_VSA6,
                                      MolSurf.SlogP_VSA7,
                                      MolSurf.SlogP_VSA8,
                                      MolSurf.SlogP_VSA9,
                                      MolSurf.SMR_VSA1,
                                      MolSurf.SMR_VSA10,
                                      MolSurf.SMR_VSA2,
                                      MolSurf.SMR_VSA3,
                                      MolSurf.SMR_VSA4,
                                      MolSurf.SMR_VSA5,
                                      MolSurf.SMR_VSA6,
                                      MolSurf.SMR_VSA7,
                                      MolSurf.SMR_VSA8,
                                      MolSurf.SMR_VSA9,
                                      MolSurf.TPSA])


        descriptor = []
        for descriptor_function in descriptor_functions:
            try:
                desc_val = descriptor_function(mol)
            except:
                continue
            if np.isnan(desc_val):
                desc_val = 0
            descriptor.append(desc_val)

        descriptor = np.array(descriptor)

        a = sum(np.isnan(descriptor))
        b = sum(np.isinf(descriptor))
        if a > 0 or b > 0:
            print(descriptor)
            exit()
        return descriptor




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
