import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.PandasTools import LoadSDF
from smorgasbord.descriptor import DatasetDescriptorCalculator
from smorgasbord.sampling import DatasetSampler
import hashlib
import copy
import os

unit_convert_dict = {
    "f": 10 ** -6,
    "p": 10 ** -3,
    "n": 10 ** 0,
    "u": 10 ** 3,
    "m": 10 ** 6,
    "c": 10 ** 7,
    "d": 10 ** 8,
    "": 10 ** 9,
    "k": 10 ** 12
}


class BaseDataset:
    def __init__(self, filepath, delimiter=None, smiles_col=1, file_hash=None):
        self.stored_args = {"filepath": filepath, "delimiter": delimiter, "smiles_col": smiles_col}

        # prep file
        self.name = os.path.basename(filepath).split(".")[0]
        self.filepath = filepath.strip()

        # check if hash matches
        self._check_hash(file_hash)

        # load file
        self._load_file(delimiter)
        self._smiles_col = self._get_column_name(smiles_col)

        # prep descriptor generation
        self.descriptor = DatasetDescriptorCalculator(cache=True)
        self.random_state = np.random.randint(1e8)
        self._failed = {}  # hold index of entries that failed to load or process

    def _generate_rdkit_mols(self):
        if "ROMol" not in self.dataset.columns:
            if self._smiles_col is not None:
                try:
                    self.dataset["ROMol"] = self.dataset[self._smiles_col].apply(
                        Chem.MolFromSmiles
                    )
                except TypeError:
                    raise ValueError(
                        f"failed to parse smiles_col at index location {self._smiles_col} verify "
                        f"that this column contains valid smile strings"
                    )
            else:
                raise ValueError(
                    f"failed to recognize smiles_col of {self._smiles_col}. it is required when not using a sdf"
                    f" file as input. if you dataset has Inchi keys or SELFIES, use the relevant qsar_util "
                    f"functions to preprocess your dataset and add SMILES"
                )

        if self.dataset["ROMol"].isna().sum() > 0:
            failed_indices = self.dataset['ROMol'].isna()

            for x in self.dataset[failed_indices].index:
                self._failed.setdefault(x, []).append("Failed to read into RDKit")

    def _check_hash(self, file_hash):
        if file_hash:
            return None
        with open(self.filepath, 'rb') as f:
            s = f.read()
            f.close()
        # TODO make sure hashing is constant
        self.file_hash = hashlib.sha256(s).hexdigest()

        if file_hash and file_hash != self.file_hash:
            raise Exception(f"File contents (sha256 hash) for {self.filepath} have changed! set check_file_contents to "
                            f"false to override")

    def _load_file(self, delimiter):
        try:
            tail = self.filepath.split(".")[1]
        except IndexError:
            raise ValueError("file type not defined please specify file type (.sdf, .csv, .tab, etc...)")

        if tail.lower() == "sdf":
            df = LoadSDF(self.filepath)

        else:
            if delimiter is not None:
                if tail == ".csv":
                    delimiter = ","
                if tail == ".tab":
                    delimiter = "\t"
                if tail == ".txt":
                    delimiter = " "
            df = pd.read_csv(self.filepath, delimiter=delimiter)

        self.dataset = df

    def get_dataset(self):
        return self.dataset.drop(index=self._failed.keys())

    def get_descriptor(self, desc_name, **kwargs):
        return np.delete(self.descriptor.get_descriptor_value(desc_name, self.dataset, **kwargs),
                         list(self._failed.keys()), axis=0)

    def _get_column_name(self, index):
        if index is not None:
            if isinstance(index, int):
                if index < len(self.dataset.columns):
                    return self.dataset.columns[index]
            elif isinstance(index, str):
                if index in list(self.dataset.columns):
                    return index
        return None

    def _get_column_names(self, indexes):
        return [self._get_column_name(idx) for idx in indexes]

    # TODO this needs to be reworked its not going to work like this and I think it should move to the desc calc object
    # def merge_descriptors(self, descriptors):
    #     name = []
    #     desc_sets = []
    #     desc_options = ()
    #     if isinstance(descriptors, str):
    #         descriptors = [descriptors]
    #     for desc in descriptors:
    #         if desc in self.descriptor.__dict__.keys():
    #             name.append(desc)
    #             desc_options = desc_options + (self.descriptor.__getattribute__(desc)[0])
    #             desc_sets.append(self.descriptor.__getattribute__(desc)[1])
    #
    #     assert len(set([_.shape for _ in desc_sets])) <= 1
    #     assert len(set([_.shape[0] for _ in desc_sets])) <= 1
    #
    #     self.add_descriptor_set("_".join(name), np.concatenate(desc_sets, axis=-1))

    def set_descriptor_func(self, name, func):
        self.descriptor.set_descriptor_function(name, func)

    # TODO need to check if that way im passing args with ** is okay
    def add_calculated_descriptor(self, name, desc, args=None):
        desc = np.array(desc).astype(float)
        if desc.shape[0] == self.dataset.shape[0]:
            if args is None: args = {}
            self.descriptor.add_calculated_descriptor(name, desc, **args)
        else:
            raise ValueError(f"descriptor dimensions does not match dataset dimension on axis 0, f{desc.shape}, "
                             f"f{self.dataset.shape}")


class ScreeningDataset(BaseDataset):
    def __init__(self, filepath, delimiter=None, smiles_col=1, curation="default", **kwargs):
        super().__init__(filepath, delimiter, smiles_col)

        self._curation = curation

    # TODO ask josh and kat if we think there are some function that are unique to screening datasets that we want


class QSARDataset(BaseDataset):
    def __init__(self, filepath, delimiter=None, curation="default", label_col=-1, smiles_col=1, label="auto",
                 cutoff=None, unit_col=None, file_hash=None):
        """
        Parameters
        ---------------------
            filepath: (str)
                the absolute or cw location of your dataset file
            delimiter: (str) | optional: default = None
                the delimiter used in the dataset file
                not used for sdf files and inferred as ",", "\\t", "\\s" for .csv, .tab, .txt files respectively
                if not a .sdf, .tab, .csv, or .txt will fail without a passed delimiter
            label_col: (str or int) | optional: default = -1
                the col name or col index of the row that contains the label for the dataset
            smiles_col: (str or int) | optional: default = -1
                the col name or col index of the row that contains the smiles strings for the dataset
                only used if the file type is not .sdf
            label: (str: "continuous", "binary", "multiclass", "auto") | optional: default = "auto"
                the input style of descriptors. When set to auto program will attempt to detect the label style
            cutoff: (float or list of floats) | optional: default = None
                if desired label is not None, will attempt to us the cutoff to convert to the desired labels. otherwise,
                it will be ignored
                binary requires 1 cutoff and will use only the first cutoff in the list if a list is given
                multiclass require n-1 cutoffs for n class and will fail if not given this
                if left as None, but required will default to making each class equal in size (equal as possible)
            curation: (func, "default" or None) | optional: default = "default"
                what type of curation to use, if any
                "default" will use chembl pipeline curation
                None will skip curation
                passing a func will result in the program using that func to do curation. see curation documentation for
                guidelines on this function
            unit_col: (str or int) | optional: default = None
                the column name or index of the units for any continuous label so that this values can be standardized
                units must be in the metric system
                ignored for non-continuous data
                if left as None assumes all continuous labels are the same unit
            file_hash: (str) | optional: default = None
                sha256 hash of input file contents
                if provided, will raise Exception on __init__() if file contents have changed

        """

        super().__init__(filepath, delimiter, smiles_col, file_hash)
        self.stored_args.update({"curation": curation, "label_col": label_col, "label": label, "cutoff": cutoff,
                                 "unit_col": unit_col})

        self._desired_label = label
        self._labels = {}
        self._curation = curation

        self._binary_cutoff = None
        self._multiclass_cutoff = None

        # sampling mask dictionary
        self.sampler = DatasetSampler(cache=True)

        ### HANDLE LABEL CLASSIFICATION ###
        if label not in ["auto", "binary", "multiclass", "continuous"]:
            raise ValueError(f"QSARDataset must have label set to 'auto', 'binary', 'multiclass' or 'continuous', "
                             f"was set to {label}")

        ### HANDLE CURATION SETTINGS ###
        if curation not in ["default", None] and not callable(curation):
            raise ValueError(f'QSARDataset must have curation set to None or "default" or be a valid curation function '
                             f'was set to {curation}')

        ### COVERT COLUMN INDEX TO COLUMN NAMES AND CHECK###
        self._label_col = self._get_column_name(label_col)
        self._unit_col = self._get_column_name(unit_col)

        # TODO lol the above wont work if a list of columns is passed, need to get the get_column_name to handle that

        if self._label_col is None:
            raise ValueError(f"Failed to recognize label_col of {label_col}. label_col is required")

        ### DETECT DATA STYLE OF LABEL COLUMN ###

        # TODO add support for multitask datasets

        if self._desired_label == "auto":
            _labels = self.dataset[self._label_col].to_list()
            _labels = list(map(float, _labels))

            _valid_labels = [x for x in _labels if not np.isnan(x)]

            if all([x.is_integer() for x in _valid_labels]):
                if len(set(_valid_labels)) == 2:
                    guess_initial_label_class = "binary"
                else:
                    guess_initial_label_class = "class_int"
            else:
                guess_initial_label_class = "continuous"

            if guess_initial_label_class == "class_int":
                count = len(set(_labels))
                # TODO think about threshold
                if count > (0.5 * len(self.dataset)):
                    guess_initial_label_class = "continuous"
                elif count > 2:
                    guess_initial_label_class = "multiclass"
                else:
                    guess_initial_label_class = "binary"

            self._desired_label = guess_initial_label_class

        # We should never get there but whatever
        # for i, label in zip(self.dataset.index, self.dataset[self._label_col]):
        #     try:
        #         float(label)
        #     except ValueError:
        #         self._failed.setdefault(i, []).append("Missing/improper initial label")
        #         self.dataset[self._label_col][i] = np.nan

        if self.dataset[self._label_col].isnull().all():
            raise ValueError("All labels corrupted. If labels are non-numeric convert to numeric with data_utils func")

        # save the initial label to the dictionary
        if self._desired_label == "continuous":
            self._labels[self._desired_label] = self.dataset[self._label_col].astype(float)
        else:
            self._labels[self._desired_label] = self.dataset[self._label_col].astype(int)

        ### The following only needs to occur is the data is currently continuous
        if self._desired_label == "continuous":

            ### Convert units ###
            if self._unit_col is not None:
                units = [unit_convert_dict[x[0]] for x in self.dataset[self._unit_col]]
                self._labels[self._desired_label] = self._labels[self._desired_label] * units

        if curation is None:
            pass
        else:
            self.curate()

    def get_dataset(self, mask_name=None):
        """
        takes the original dataset and drops failed indices (failed rdkit mols, missing labels, etc)
        josh will always use this to access the dataframe and leave the original dataframe alone
        """
        mask = self._union_failed_mask(mask_name) if mask_name is not None else list(self._failed.keys())
        return self.dataset.drop(index=mask)

    def get_descriptor(self, desc_name, mask_name=None, **kwargs):
        mask = self._union_failed_mask(mask_name) if mask_name is not None else list(self._failed.keys())
        return np.delete(self.descriptor.get_descriptor_value(desc_name, self.dataset, **kwargs), mask, axis=0)

    def get_label(self, mask_name=None):
        mask = self._union_failed_mask(mask_name) if mask_name is not None else list(self._failed.keys())
        return np.delete(self.get_labels(self._desired_label), mask, axis=0)

    def get_labels(self, kind):
        """
        takes the private _labels and drops failed indices (failed rdkit mols, missing labels, etc)
        josh will always use this to access the labels and leave the original labels alone
        """
        if kind == "binary" or kind == "multiclass":
            dtype = int
        elif kind == "continuous":
            dtype = float
        else:
            raise Exception(f"Supplied kind {kind} not implemented in get_labels()")

        return np.array(self._labels[kind].drop(index=self._failed.keys()), dtype=dtype)

    def set_label(self, name, label):
        label = np.array(label)
        if label.shape[0] != self.dataset.shape[0] or len(label.shape) > 2:
            raise ValueError(f"label is malformed got shape {label.shape} excepted ({self.dataset.shape[0]}, 1)")
        self._labels[name] = label

    def set_desired_label(self, label_name):
        if label_name in self.get_existing_labels():
            self._desired_label = label_name
        else:
            raise ValueError(f"label {label_name} does not exist")

    def set_desired_label_to_binary(self):
        if self.has_binary_label():
            self._desired_label = "binary"
        else:
            raise ValueError("binary label does not exist for dataset")

    def get_desired_label(self):
        return self._desired_label

    def get_existing_labels(self):
        return list(self._labels.keys())

    def iter_labels(self):
        for key, val in self._labels.items():
            yield key, val

    def has_binary_label(self):
        return "binary" in self._labels.keys()

    def to_binary(self, cutoff=None, class_names=None, return_cloned_df=False):
        """
        covert the continuous column of the dataset to a multiclass label.
        Defaults to saving as label in the dataset but could return a clone of the dataset with just
        this label if requested

        will override any existing binary label

        will also set binary label as active. active label can be changed using set_active_label function and all
        existing labels can be viewed with get_labels

        Parameters
        ------------------------------------
            cutoff: float or list(float) | optional default = None
                the locations to make the class splits. ei [2] would make the two class, one < 2, and one >= 2
                >5. If left as None will default to using num class or class names to determine number of requested
                classes and will attempt to make every class equal in size
            class_names: (list): | optional default = None
                the names of the new classes. Is order matched to cutoff. If left as None names default to 0-n where
                n is number of classes. If not None, must match is length any other non None parameter
            return_cloned_df: (bool) | optional default = False
                instead of assigning this label to the current dataframe object and making it the active label, will
                return a cloned copy of the dataset object with this label set to active, leaving the original dataset
                object unchanged

        Returns
        -----------------------------------
            new_dataset (QSARDataset) | optional
                will only return the new QSARDataset object if return_cloned_df set to True
        """
        cutoff = self._check_cutoff(cutoff)
        if cutoff is None:
            cutoff = [np.median(self._labels["continuous"][self._labels["continuous"].notna()])]
        elif len(cutoff) > 1:
            cutoff = [cutoff[0]]
        self._binary_cutoff = cutoff
        if return_cloned_df:
            raise NotImplemented
            # TODO make the object clone able and return the new dataset object with just this label
        self._labels["binary"] = self._create_discrete_labels(cutoff=cutoff, class_names=class_names)

    def set_binary_label(self, label):
        if len(set(label)) != 2:
            raise ValueError(f"binary label not binary. got {len(set(label))} unique classes expected 2")
        self.set_label("binary", label)

    def get_binary_label(self):
        if "binary" in self._labels.keys():
            return self._labels["binary"]
        else:
            raise ValueError("dataset has no binary label. Try to create one with to_binary")

    def has_multiclass_label(self):
        return "multiclass" in self._labels.keys()

    def to_multiclass(self, cutoff=None, num_classes=None, class_names=None, return_cloned_df=False):
        """
        covert the continuous column of the dataset to a binary label.
        Defaults to saving as label in the dataset but could return a clone of the dataset with just
        this label if requested

        will override any existing binary label

        will also set binary label as active. active label can be changed using set_active_label function and all
        existing labels can be viewed with get_labels

        Parameters
        ------------------------------------
            cutoff: float or list(float) | optional default = None
                the locations to make the class splits. ei [2,5] would make 3 class, one below 2, one from (2, 5] and
                >5. If left as None will default to using num class or class names to determine number of requested
                classes and will attempt to make every class equal in size
            num_classes: (int) | optional default = None
                the number of desired classes to make. If left as None number of classes will be assumed from cutoff or
                class names. If not None, must match is length any other non None parameter
            class_names: (list): | optional default = None
                the names of the new classes. Is order matched to cutoff. If left as None names default to 0-n where
                n is number of classes. If not None, must match is length any other non None parameter
            return_cloned_df: (bool) | optional default = False
                instead of assigning this label to the current dataframe object and making it the active label, will
                return a cloned copy of the dataset object with this label set to active, leaving the original dataset
                object unchanged

        Returns
        -----------------------------------
            new_dataset (QSARDataset) | optional
                will only return the new QSARDataset object if return_cloned_df set to True
        """
        cutoff = self._check_cutoff(cutoff)
        if num_classes is not None and num_classes < 2:
            raise ValueError(f"must have at least 2 classes passed {num_classes}")
        if class_names is not None and (num_classes is not None and len(num_classes) != len(class_names)):
            raise ValueError("class names must match number of classes")
        if num_classes is None and class_names is not None:
            if len(class_names) > 1:
                num_classes = len(class_names)
            else:
                raise ValueError(f"number of classes must be at least 2 found {len(class_names)}")
        if cutoff is None:
            if num_classes is None:
                raise ValueError("if cutoff is None num_classes must not be None and greater than 1")
            else:
                cutoff = np.quantile(self._labels["continuous"], np.arange(1 / num_classes, 1, 1 / num_classes))
        self._multiclass_cutoff = cutoff
        if return_cloned_df:
            new_dataset = copy.deepcopy(self)
            new_dataset.__setattr__("_labels", self._create_discrete_labels(cutoff=cutoff, class_names=class_names))
        self._labels["multiclass"] = self._create_discrete_labels(cutoff=cutoff, class_names=class_names)

    def set_multiclass_label(self, label):
        self.set_label("multiclass", label)

    def get_multiclass_label(self):
        if "multiclass" in self._labels.keys():
            return self._labels["multiclass"]
        else:
            raise ValueError("dataset has no multiclass label. Try to create one with to_multiclass")

    def _create_discrete_labels(self, cutoff, class_names):
        if class_names is not None:
            if len(class_names) != len(cutoff) + 1:
                raise ValueError(f"number of class names and number of cutoffs mismatches. got {len(cutoff) + 1} class "
                                 f"and {len(class_names)} class names")
        else:
            class_names = np.arange(0, len(cutoff) + 1)

        cutoff = [min(self._labels["continuous"].astype(float)) - 1] + list(cutoff) + \
                 [max(self._labels["continuous"].astype(float)) + 1]
        labels = np.full(self._labels["continuous"].shape[0], "")

        for i in range(len(class_names)):
            labels[self._labels["continuous"].astype(float).between(cutoff[i], cutoff[i + 1], inclusive="right")] = \
                class_names[i]
        return pd.Series(labels, index=self._labels["continuous"].index)

    @staticmethod
    def _check_cutoff(cutoff):
        if cutoff is not None:
            try:
                iter(cutoff)
                cutoff = list(cutoff)
            except TypeError:
                cutoff = [cutoff]

            if not all([not isinstance(x, str) for x in cutoff]):
                raise ValueError(f"cutoff values need to be numeric, not chars. found {cutoff} "
                                 f"of type {[type(x) for x in cutoff]}")
            return cutoff
        else:
            return None

    def scale_label(self, label, scale_factor):
        scaled_label_name = f"scaled_{scale_factor}"
        if isinstance(label, str):
            scaled_label_name = f"{label}_{scaled_label_name}"

        if label in self._labels.keys():
            label = self._labels[label]

        label = np.array(label) * scale_factor

        self.set_label(scaled_label_name, label)

    def normalize_label(self, label):
        normalized_label_name = f"normalized"
        if isinstance(label, str):
            normalized_label_name = f"{label}_{normalized_label_name}"

        if label in self._labels.keys():
            label = self._labels[label]

        label = np.array(label) / max(label)

        self.set_label(normalized_label_name, label)

    def to_dict(self):
        return {"Arguments": self.stored_args,
                "Name": self.name,
                "Filepath": self.filepath,
                "Label Column": self._label_col,
                "SMILES Column": self._smiles_col,
                "Label": self._desired_label,
                "File Hash": self.file_hash}

    def curate(self):
        from curate import curate_mol
        results = [curate_mol(x) for x in self.dataset["ROMol"]]

        passed = [x[1].passed for x in results]
        curated_mols = [x[0] for x in results]
        histories = [x[1] for x in results]
        modified = [x[1].structure_modified for x in results]

        failed = [not x[1].passed for x in results]
        new_fail_dict = {x: "Did not pass automatic curation" for x in self.dataset[failed].index}
        self._failed.update(new_fail_dict)

        self.dataset["Curation history"] = histories
        self.dataset["Passed curation"] = passed
        self.dataset["Curation modified structure"] = modified
        self.dataset["Curated ROMol"] = curated_mols

    # wrapper function (helpful for readability). If you want the actual balanced dataset should use get_dataset(mask)
    def balance(self, method="downsample"):
        if method is not None:
            self._get_balance_indices(method)

    def _get_balance_indices(self, method):
        return self.sampler.get_mask(method, self.dataset, self.get_labels(self.get_desired_label()))

    def _union_failed_mask(self, mask_name):
        return list(set(list(self._get_balance_indices(mask_name)) + list(self._failed.keys())))
