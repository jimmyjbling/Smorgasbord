import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.PandasTools import LoadSDF
from collections import Counter
from descriptor import DescriptorCalculator

unit_covert_dict = {
    "f": 10**-6,
    "p": 10**-3,
    "n": 10**0,
    "u": 10**3,
    "m": 10**6,
    "c": 10**7,
    "d": 10**8,
    "": 10**9,
    "k": 10**12
}

# TODO might be better to abstract this to a base dataset class and then a GraphData set child and a QSAR child

# TODO add classes for torch dataset objects and pytorch geometric datasets???

# TODO make dataset iterable so that it can be used with torch data loaders


class QSARDataset:
    def __init__(self, filepath, delimiter=None, curation="default", label_col=-1, smiles_col=1, label="auto",
                 desired_label=None, cutoff=None,  unit_col=None, mixture=False, mixture_columns=None):
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

        the dataset can be curated to standardize chemicals and remove mixture and duplicates using the chembl curation
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
        possible descriptors and their arguments. Once generated, descriptors will be cached so they will only need to
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
            desired_label: (str: "continuous", "binary", None) | optional: default = None
                if label is continuous, convert the label to binary or multiclass using the cutoff parameter
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
        """

        # TODO add support for mixtures splits

        self.label = label
        self._active_label = label
        self.desired_label = desired_label
        self.curation = curation
        self.mixture = mixture

        self._failed = []

        self.labels = {}
        self.masks = {}
        self.descriptor = DescriptorCalculator(cache=True)

        self.children = {}

        self._random_state = np.random.randint(1e8)
        # TODO set a random state on init so that results are repeatable if wanted

        # TODO store the binary cutoff and multi class cutoff of the model so the exact settings and parameters can be
        #  saved into a jsons file for replication

        self._binary_cutoff = None
        self._multiclass_cutoff = None

        ### HANDLE LABEL CLASSIFICATION ###
        if label not in ["auto", "binary", "multiclass", "continuous"]:
            raise ValueError(f"QSARDataset must have label set to 'auto', 'binary', 'multiclass' or 'continuous', "
                             f"was set to {label}")

        ### HANDLE DESIRED LABEL CLASSIFICATION ###
        if desired_label not in [None, "binary", "multiclass", "continuous"]:
            raise ValueError(f"QSARDataset must have desired_label set to None, 'binary', 'multiclass' or "
                             f"'continuous', was set to {desired_label}")

        ### MAKE SURE DESIRED LABEL IS POSSIBLE ###
        if desired_label is not None:
            if label in ["binary", "multiclass"] and desired_label != label:
                raise ValueError(f"can can only convert from continuous to binary/multiclass, set to go from {label} "
                                 f"to {desired_label}")

        # ### HANDLE BALANCING SETTINGS ###
        # if balancing not in [None, "oversample", "downsample", "remove_similar", "remove_dissimilar"]:
        #     raise ValueError(f'QSARDataset must have balancing set to None, "oversample", "downsample", '
        #                      f'"remove_similar" or "remove_dissimilar" was set to {balancing}')

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

        if tail == ".sdf":
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

        self.dataset = df

        ### COVERT COLUMN INDEX TO COLUMN NAMES AND CHECK###
        self._label_col = self._get_column_name(label_col)
        self._smiles_col = self._get_column_name(smiles_col)
        self._unit_col = self._get_column_name(unit_col)
        self._mixture_cols = self._get_column_name(mixture_columns) if isinstance(mixture_columns, str) \
            else [self._get_column_name(_) for _ in mixture_columns]

        if self._label_col is None:
            raise ValueError(f"Failed to recognize label_col of {label_col}. label_col is required")

        ### DETECT DATA STYLE OF LABEL COLUMN ###

        labels = self.dataset[self._label_col]

        # TODO add support for multi label datasets

        if self.label == "auto":
            _labels = labels.to_list()
            try:
                _labels = list(map(float, _labels))

                if all([x.is_integer() for x in _labels]):
                    guess_initial_label_class = "continuous"
                else:
                    guess_initial_label_class = "class_int"
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

            self.label = guess_initial_label_class
            self._active_label = guess_initial_label_class

        # save the initial labels to the dictionary
        if self.label == "continuous":
            self.labels[self.label] = labels.astype(float)
        else:
            self.labels[self.label] = labels

        ### make an ROMol column if not already present ###
        if "ROMol" not in self.dataset.columns:
            if self._smiles_col is not None:
                try:
                    self.dataset["ROMol"] = self.dataset[self.dataset.columns[self._smiles_col]].apply(Chem.MolFromSmiles)
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
            self.failed = self.dataset[self.dataset['ROMol'].isna()]
            self.dataset.dropna(subset=['ROMol', ''])

        if curation is None:
            pass
        else:
            # TODO add support for curation
            raise NotImplemented("lol someone make me do this")

        ### The following only needs to occur is the data is currently continuous
        if self.label == "continuous":

            ### Convert units ###
            if self._unit_col is not None:
                units = [unit_covert_dict[x[0]] for x in self.dataset[self._unit_col]]
                self.labels[self.label] = self.labels[self.label] * units

            ### convert to desired label ###
            if self.label != self.desired_label and self.desired_label is not None:
                if self.desired_label == "binary":
                    self.to_binary(cutoff)
                if self.desired_label == "multiclass":
                    self.to_multiclass(cutoff)

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
            cutoff: (float or list(float) | optional default = None
                the locations to make the class splits. ei [2] would make the two class, one < 2, and one >= 2
                >5. If left as None will default to using num class or class names to determine number of requested
                classes and will attempt to make every class equal in size
            class_names: (list): | optional default = None
                the names of the new classes. Is order matched to cutoff. If left as None names default to 0-n where
                n is number of classes. If not None, must match is length any other non None parameter
            return_cloned_df: (bool) | optional default = False
                instead of assigning this label to the current dataframe object and making it the active label, will
                return a cloned copy of the dataset object with this label set to active, leaving the orginal dataset
                object unchanged

        Returns
        -----------------------------------
            new_dataset (QSARDataset) | optional
                will only return the new QSARDataset object if return_cloned_df set to True
        """
        cutoff = self._check_cutoff(cutoff)
        if cutoff is None:
            cutoff = [np.median(self.labels["continuous"])]
        elif len(cutoff) > 1:
            cutoff = [cutoff[0]]
        self._binary_cutoff = cutoff
        if return_cloned_df:
            raise NotImplemented
            # TODO make the object clone able and return the new dataset object with just this label
        self.labels["binary"] = self._split_data(cutoff=cutoff, class_names=class_names)

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
            cutoff: (float or list(float) | optional default = None
                the locations to make the class splits. ei [2,5] would make 3 class, one below 2, one from [2, 5) and
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
                return a cloned copy of the dataset object with this label set to active, leaving the orginal dataset
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
                cutoff = np.quantile(self.labels["continuous"], np.arange(1 / num_classes, 1, 1 / num_classes))
        self._multiclass_cutoff = cutoff
        if return_cloned_df:
            raise NotImplemented
            # TODO make the object clone able and return the new dataset object with just this label
        self.labels["multiclass"] = self._split_data(cutoff=cutoff, class_names=class_names)

    def _split_data(self, cutoff, class_names):
        if class_names is not None:
            if len(class_names) != len(cutoff)+1:
                raise ValueError(f"number of class names and number of cutoffs mismatches. got {len(cutoff) + 1} class "
                                 f"and {len(class_names)} class names")
        else:
            class_names = np.arange(0, len(cutoff)+1)

        cutoff = [0] + list(cutoff) + [1]
        labels = np.full(len(class_names), "")
        for i in range(len(class_names)):
            labels[self.labels["continuous"].astype(float).between(cutoff[i], cutoff[i+1], inclusive="left")] = class_names[0]
        return pd.Series(labels, index=self.labels["continuous"].index)

    @staticmethod
    def _check_cutoff(cutoff):
        if cutoff is not None:
            try:
                iter(cutoff)
                cutoff = list(cutoff)
            except TypeError:
                cutoff = [cutoff]

            if all([not isinstance(x, str) for x in cutoff]):
                raise ValueError(f"cutoff values need to be numeric, not chars. found {cutoff} "
                                 f"of type {[type(x) for x in cutoff]}")
            return cutoff
        else:
            return []

    def scale_label(self, scale_factor):
        raise NotImplemented
        # TODO implement function

    def normalize_label(self, norm):
        raise NotImplemented
        # TODO implement function

    def normalize_descriptors(self, norm):
        raise NotImplemented
        # TODO implement function

    def _get_column_name(self, index):
        if index is not None:
            if isinstance(index, int):
                if index < len(self.dataset.columns):
                    return self.dataset.columns[index]
            elif isinstance(index, str):
                if index in list(self.dataset.columns):
                    return index
        return None

    @property
    def active_label(self):
        return self._active_label

    @active_label.setter
    def active_label(self, value):
        if value not in self.labels.keys():
            raise ValueError("label does not exist")
        else:
            self._active_label = value

    def get_children(self):
        return list(self.children.keys())

    def add_child(self, name, indices):
        if name in self.children.keys():
            # TODO add in warning about overriding?
            raise RuntimeWarning("child already exists overwriting")
        self.children[name] = indices

    def remove_child(self, name):
        if name in self.children.keys():
            del self.children[name]
        else:
            # TODO maybe just logging?
            raise RuntimeWarning("child does not exist cannot delete")

    def model(self, model, name=None):
        raise NotImplemented

    def get_models(self, name=None):
        raise NotImplemented

    def remove_model(self, name):
        raise NotImplemented

    def screen(self, screening_dataset, model=None):
        raise NotImplemented

    def balance(self, method="downsample"):
        raise NotImplemented

    def cross_validate(self, cv, model=None):
        raise NotImplemented



