import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.PandasTools import LoadSDF
from collections import Counter
from descriptor import DescriptorCalculator
import hashlib
import copy
import os

unit_covert_dict = {
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


# TODO might be better to abstract this to a base dataset class and then a GraphData set child and a QSAR child

# TODO add classes for torch dataset objects and pytorch geometric datasets???

# TODO make dataset iterable so that it can be used with torch data loaders

# TODO someday add support for a given dataset to be passed rather than a file to be read in

# TODO QUESTION: should we support datasets that do not have a molecule associated with them???


class QSARDataset:
    def __init__(self, filepath, delimiter=None, curation="default", label_col=-1, smiles_col=1, label="auto",
                 cutoff=None, unit_col=None, file_hash=None):
        """
        base dataset object that will handle curation and process of molecular datasets. dataset objects can handle
        the following actions to process your dataset:
            - Generating rdkit molecule objects for your chemicals
            - Curation and standardization of your chemicals
            - Standardization and conversion of a continuous label to a classification label based on a cutoff
            - Generating any of the descriptors for your datasets
            - Balancing your datasets in a number of different methods

        dataset should contain some description of the molecule (.sdf or smiles) and a label for each chemical
        the label can be continuous, binary or multiclass. If not told explicitly the program will attempt to infer
        the label, but this could result in incorrect assessment so optimal behavior would be to directly set this

        the dataset can be curated to standardize chemicals and remove mixture and duplicates using the CHEMBL curation
        pipeline. Additionally, you can pass a custom curation function that takes in a list of rdkit mols and
        returns a curated list of rdkit mols. You can find guidelines for this function in the curation documentation

        If the label is continuous, and you want to move to a binary label or multiclass label, you can set desired
        label to either binary or multiclass. In this case you can also pass a cutoff, which for binary should be a
        single float, and multiclass a list of floats. If no cutoff is directly set the program will default to "auto"
        mode and pick a cutoff that makes each class have as equal weight as possible in the dataset. Binary and
        multiclass labels can be created later as well, using the .to_binary() and to_multiclass() functions. Note that
        the initial continuous label will not be removed or deleted but stored incase it needs to be referenced again.
        when creating multiclass or binary labels, if new ones are created the previous will be overwritten. If multiple
        binary cutoffs for a give dataset are required clone the dataset object and give each clone a different cutoff.
        when modeling you can set the active label to use by the dataset by using the set_active_label() function. you
        can reference what it is using the get_active_label() function. By default, the active label will change to the
        last time of label generated, of the initial label if no new label is ever created. You cannot create a binary
        or multiclass label if the original dataset never contained a continuous label, as there is no way to make such
        a conversion.

        You create descriptors for your dataset by calling the relevant descriptor set name on the dataset. for example,
        dataset.rdkit(). You can also pass the generation parameters if they exist. for example dataset.morgan(3, 1024)
        get you the morgan fingerprints of radius 3 nbits 1024. You can reference the descriptor documentation for
        possible descriptors and their arguments. Once generated, descriptors will be cached, so they will only need to
        be generated once for a given argument setting. If you make the same call to the same descriptor set again it
        will reference this cache to save time.

        children of the dataset set can be created as well with the make_child() function. when passed a name and list
        of dataset index it will save a subset of the dataset with only those entries. Children can be reference by
        calling their name on the dataset, like dataset.child1. Child names can not contain any whitespace chars

        balancing of the dataset can also be done in a various ways. Balancing will not alter the dataset in any way,
        rather make a child of the dataset named after the balancing method, unless told not to do this

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

        self.stored_args = {"filepath": filepath, "delimiter": delimiter, "curation": curation,
                            "label_col": label_col, "smiles_col": smiles_col, "label": label,
                            "cutoff": cutoff, "unit_col": unit_col}

        # TODO add support for mixtures splits

        self.name = os.path.basename(filepath)

        self._label = label
        self._labels = {}
        self._curation = curation

        self._failed = {}

        self.descriptor = DescriptorCalculator(cache=True)

        self._masks = {}

        with open(filepath, 'rb') as f:
            s = f.read()
            f.close()
        self.file_hash = hashlib.sha256(s).hexdigest()

        if file_hash and file_hash != self.file_hash:
            raise Exception(f"File contents (sha256 hash) for {filepath} have changed! set check_file_contents to "
                            f"false to override")

        self._models = {}
        self._cv = {}

        self._random_state = np.random.randint(1e8)

        self._binary_cutoff = None
        self._multiclass_cutoff = None

        ### HANDLE LABEL CLASSIFICATION ###
        if label not in ["auto", "binary", "multiclass", "continuous"]:
            raise ValueError(f"QSARDataset must have label set to 'auto', 'binary', 'multiclass' or 'continuous', "
                             f"was set to {label}")

        ### HANDLE CURATION SETTINGS ###
        if curation not in ["default", None] and not callable(curation):
            raise ValueError(f'QSARDataset must have curation set to None or "default" or be a valid curation function '
                             f'was set to {curation}')

        ### LOAD IN THE DATA AND GET LABELS AND RDKIT MOLS ###
        # TODO add the ability to read in lists of files and mol file support

        self.filepath = filepath.strip()

        try:
            tail = self.filepath.split(".")[1]
        except IndexError:
            raise ValueError("file type not defined please specify file type (.sdf, .csv, .tab, etc...)")

        if tail.lower() == "sdf":
            df = LoadSDF(filepath)

        else:
            if delimiter is not None:
                if tail == ".csv":
                    delimiter = ","
                if tail == ".tab":
                    delimiter = "\t"
                if tail == ".txt":
                    delimiter = " "
            df = pd.read_csv(filepath, delimiter=delimiter)

        # note dataset will always be indexed from 0 ... n if it is not we have a problem
        self.dataset = df

        ### COVERT COLUMN INDEX TO COLUMN NAMES AND CHECK###
        self._label_col = self._get_column_name(label_col)
        self._smiles_col = self._get_column_name(smiles_col)
        self._unit_col = self._get_column_name(unit_col)

        # TODO lol the above wont work if a list of columns is passed, need to get the get_column_name to handle that

        if self._label_col is None:
            raise ValueError(f"Failed to recognize label_col of {label_col}. label_col is required")

        ### DETECT DATA STYLE OF LABEL COLUMN ###

        # TODO add support for multi label datasets

        if self._label == "auto":
            _labels = self.dataset[self._label_col].to_list()
            try:
                _labels = list(map(float, _labels))

                _valid_labels = [x for x in _labels if not np.isnan(x)]

                if all([x.is_integer() for x in _valid_labels]):
                    if len(set(_valid_labels)) == 2:
                        guess_initial_label_class = "binary"
                    else:
                        guess_initial_label_class = "class_int"
                else:
                    guess_initial_label_class = "continuous"
            except ValueError:
                guess_initial_label_class = "class"

            if guess_initial_label_class in ["class" or "class_int"]:
                count = len(Counter(_labels))
                if count > (0.8 * len(self.dataset)) and guess_initial_label_class == "class_int":
                    guess_initial_label_class = "continuous"
                elif count > 2:
                    guess_initial_label_class = "multiclass"
                else:
                    guess_initial_label_class = "binary"

            self._label = guess_initial_label_class

        for i, label in zip(self.dataset.index, self.dataset[self._label_col]):
            try:
                float(label)
            except ValueError:
                self._failed.setdefault(i, []).append("Missing/improper initial label")
                self.dataset[self._label_col][i] = np.nan

        if self.dataset[self._label_col].isnull().all():
            raise ValueError("All labels corrupted. If labels are non-numeric convert to numeric with data_utils func")

        # save the initial labels to the dictionary
        if self._label == "continuous":
            self._labels[self._label] = self.dataset[self._label_col].astype(float)
        else:
            self._labels[self._label] = self.dataset[self._label_col].astype(int)

        ### make an ROMol column if not already present ###
        if "ROMol" not in self.dataset.columns:
            # TODO someday add support for inchi code and SELFIES to be the default too, not just smiles
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
                    f"failed to recognize smiles_col of {smiles_col}. it is required when not using a sdf"
                    f" file as input. if you dataset has Inchi keys or SELFIES, use the relevant qsar_util "
                    f"functions to preprocess your dataset and add SMILES"
                )

        if self.dataset["ROMol"].isna().sum() > 0:
            failed_indices = self.dataset['ROMol'].isna()

            for x in self.dataset[failed_indices].index:
                self._failed.setdefault(x, []).append("Failed to read into RDKit")

        if curation is None:
            pass
        else:
            self.curate()

        ### The following only needs to occur is the data is currently continuous
        if self._label == "continuous":

            ### Convert units ###
            if self._unit_col is not None:
                units = [unit_covert_dict[x[0]] for x in self.dataset[self._unit_col]]
                self._labels[self._label] = self._labels[self._label] * units

    def get_dataset(self):
        """
        takes the original dataset and drops failed indices (failed rdkit mols, missing labels, etc)
        josh will always use this to access the dataframe and leave the original dataframe alone
        """
        return self.dataset.drop(index=self._failed.keys())

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

    def add_label(self, name, label):
        if name in self._labels.keys():
            raise ValueError(f"label {name} already exists, use set_label to override")
        self.set_label(name, label)

    def set_label(self, name, label):
        label = np.array(label)
        if label.shape[0] != self.dataset.shape[0] or len(label.shape) > 2:
            raise ValueError(f"label is malformed got shape {label.shape} excepted ({self.dataset.shape[0]}, 1)")
        self._labels[name] = label

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
        self._labels["binary"] = self._split_data(cutoff=cutoff, class_names=class_names)

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
            new_dataset.__setattr__("_labels", self._split_data(cutoff=cutoff, class_names=class_names))
        self._labels["multiclass"] = self._split_data(cutoff=cutoff, class_names=class_names)

    def set_multiclass_label(self, label):
        self.set_label("multiclass", label)

    def get_multiclass_label(self):
        if "multiclass" in self._labels.keys():
            return self._labels["multiclass"]
        else:
            raise ValueError("dataset has no multiclass label. Try to create one with to_multiclass")

    def _split_data(self, cutoff, class_names):
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

    def _get_column_name(self, index):
        # TODO handle if index is a list of column names or indices
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

    def to_dict(self):
        return {"Arguments": self.stored_args,
                "Name": self.name,
                "Filepath": self.filepath,
                "Label Column": self._label_col,
                "SMILES Column": self._smiles_col,
                "Label": self._label,
                "File Hash": self.file_hash}

    def get_masks(self):
        return self._masks

    def get_mask(self, mask_name):
        return self._masks[mask_name]

    def get_masked_dataset(self, mask):
        if mask in self._masks.keys():
            mask = self._masks[mask]
        return self.dataset.loc[mask]

    def iter_masks(self):
        for key, val in self._masks.items():
            yield key, val

    def add_mask(self, name, indices):
        self._masks[name] = indices

    def remove_mask(self, name):
        if name in self._masks.keys():
            del self._masks[name]

    def calc_descriptor(self, name):
        func_call = "calc_" + str(name)
        if self.descriptor.func_exists(name):
            self.descriptor.__getattribute__(func_call)(self.dataset)
        else:
            raise AttributeError(f"cannot find function to calculate descriptor set {name}. can calculate "
                                 f"{[_.replace('calc_', '') for _ in dir(self.descriptor) if 'calc_' in _ and callable(self.descriptor.__getattribute__(_))]}")

    def merge_descriptors(self, descriptors):
        name = []
        desc_sets = []
        desc_options = ()
        if isinstance(descriptors, str):
            descriptors = [descriptors]
        for desc in descriptors:
            if desc in self.descriptor.__dict__.keys():
                name.append(desc)
                desc_options = desc_options + (self.descriptor.__getattribute__(desc)[0])
                desc_sets.append(self.descriptor.__getattribute__(desc)[1])

        assert len(set([_.shape for _ in desc_sets])) <= 1
        assert len(set([_.shape[0] for _ in desc_sets])) <= 1

        self.add_descriptor_set("_".join(name), np.concatenate(desc_sets, axis=-1))

    def calc_custom_descriptor(self, name, func, kwargs=None):
        self.descriptor.calc_custom(self.dataset, name, func, kwargs)

    def add_descriptor_set(self, name, desc):
        desc = np.array(desc)
        if desc.shape[0] == self.dataset.shape[0]:
            self.descriptor.add_descriptor(name, desc)

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

    def balance(self, method="downsample"):
        raise NotImplemented
        # TODO implement function
