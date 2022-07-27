import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.PandasTools import LoadSDF
from collections import Counter
from descriptor import DescriptorCalculator
import hashlib
import copy
from model import QSARModel
import os
import pprint

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

# TODO someday add support for a given dataset to be passed rather than a file to be read in

# TODO QUESTION: should we support datasets that do not have a molecule associated with them???


class QSARDataset:
    def __init__(self, filepath, delimiter=None, curation="default", label_col=-1, smiles_col=1, label="auto",
                 desired_label=None, cutoff=None,  unit_col=None, mixture=False, mixture_columns=None,
                 file_hash = None, tasks = ["classification"]):
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
            file_hash: (str) | optional: default = None
                sha256 hash of input file contents
                if provided, will raise Exception on __init__() if file contents have changed

        """

        self.stored_args = {}
        self.stored_args["filepath"] = filepath, 
        self.stored_args["delimiter"] = delimiter, 
        self.stored_args["curation"] = curation, 
        self.stored_args["label_col"] =  label_col, 
        self.stored_args["smiles_col"] = smiles_col, 
        self.stored_args["label"] = label,
        self.stored_args["desired_label"] = desired_label, 
        self.stored_args["cutoff"] = cutoff,  
        self.stored_args["unit_col"] = unit_col, 
        self.stored_args["mixture"] = mixture, 
        self.stored_args["mixture_columns"] = mixture_columns

        #TODO: check for errors here
        self.tasks = tasks

        #why is this necessary? everything ends up in a tuple?
        tmp_dict = {}
        for key, val in self.stored_args.items():
            try:
                tmp_dict[key] = val[0]
            except:
                tmp_dict[key] = val

        self.stored_args = tmp_dict

        
        # TODO add support for mixtures splits

        self.name = os.path.basename(filepath)

        self._label = label
        self._labels = {}
        #self.active_label = label
        self.curation = curation
        self.mixture = mixture

        self._failed = {}

        self.descriptor = DescriptorCalculator(cache=True)

        self._children = {}

        f = open(filepath, 'rb')
        s = f.read()
        f.close()
        hashval = hashlib.sha256(s).hexdigest()
        self.file_hash = hashval

        if file_hash:
            if file_hash != self.file_hash:
                raise Exception(f"File contents (sha256 hash) for {filepath} have changed!!!")

        self._models = {}
        self._cv={}

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
        self._mixture_cols = self._get_column_name(mixture_columns)

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
            #self.active_label = guess_initial_label_class


        #check for initial missing labels

        #TODO vectorize
        for i, label in enumerate(self.dataset[self._label_col]):
            try:
                x = float(label)
            except:
                self._failed[i] = "Missing initial label"
                self.dataset[self._label_col][i] = np.nan

            #TODO: delete this when confident it's not needed
            '''
            print(f"|{label}|")
            if label == "":
                raise Exception
            elif np.isnan(label):
            #also don't overwrite here, should keep multiple fail reasons
                self._failed[i] = "Missing initial label"

            #missing_labels =self.dataset[self._label_col].astype(float))
            #missing_labels = np.isnan(self.dataset[self._label_col].astype(float))
            #new_fail_dict = {x:"Missing initial label" for x in self.dataset[missing_labels].index}
            #self._failed.update(new_fail_dict)
            '''

        # save the initial labels to the dictionary
        if self._label == "continuous":
            self._labels[self._label] = self.dataset[self._label_col].astype(float)
        else:
            self._labels[self._label] = self.dataset[self._label_col]

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
            failed = self.dataset[self.dataset['ROMol'].isna()]


            new_fail_dict = {x:"Failed to read into RDKit" for x in self.dataset[failed_indices].index}

            self._failed.update(new_fail_dict)
            #what is the empty string for? #self.dataset.dropna(subset=['ROMol', ''])
            #self.dataset = self.dataset.dropna(subset=['ROMol'])

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

            ### convert to desired label ###
            if self._label != desired_label and desired_label is not None:
                if desired_label == "binary":
                    self.to_binary(cutoff)
                if desired_label == "multiclass":
                    self.to_multiclass(cutoff)

    def get_dataset(self):

        """
        takes the original dataset and drops failed indices (failed rdkit mols, missing labels, etc)
        josh will always use this to access the dataframe and leave the original dataframe alone
        """


        failed_indices = self._failed.keys()
        return self.dataset.drop(index = failed_indices)

    def get_labels(self, kind):
 
        """
        takes the private _labels and drops failed indices (failed rdkit mols, missing labels, etc)
        josh will always use this to access the labels and leave the original labels alone
        """

       
        failed_indices = self._failed.keys()

        if kind == "binary":
            dtype = int
        elif kind == "continuous":
            dtype = float
        else:
            raise Exception("Supplied kind ({kind}) not implemented in get_labels()")

        return np.array(self._labels[kind].drop(index = failed_indices), dtype = dtype)

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
        label = np.array(label)
        assert len(label.shape) == 1
        assert label.shape[0] == self.dataset.shape[0]
        assert len(set(label)) == 2
        self._labels["binary"] = label

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
                # TODO this will not work well if there are lots of duplicates in the dataset...
                #  but a method that fixes this would slow down this greatly on large datasets...
                #  maybe a trade of is having a check that looks for many duplicates in the dataset
                #  or just let user know that this should not be used in such a case and manual cutoffs
                #  need to be set???
                cutoff = np.quantile(self._labels["continuous"], np.arange(1 / num_classes, 1, 1 / num_classes))
        self._multiclass_cutoff = cutoff
        if return_cloned_df:
            raise NotImplemented
            # TODO make the object clone able and return the new dataset object with just this label
        self._labels["multiclass"] = self._split_data(cutoff=cutoff, class_names=class_names)

    def set_multiclass_label(self, label):
        label = np.array(label)
        assert len(label.shape) == 1
        assert label.shape[0] == self.dataset.shape[0]
        self._labels["multiclass"] = label

    def _split_data(self, cutoff, class_names):
        if class_names is not None:
            if len(class_names) != len(cutoff)+1:
                raise ValueError(f"number of class names and number of cutoffs mismatches. got {len(cutoff) + 1} class "
                                 f"and {len(class_names)} class names")
        else:
            class_names = np.arange(0, len(cutoff)+1)

        cutoff = [min(self._labels["continuous"].astype(float)) - 1] + list(cutoff) + \
                 [max(self._labels["continuous"].astype(float)) + 1]
        labels = np.full(self._labels["continuous"].shape[0], "")
        for i in range(len(class_names)):
            # lol sorry I had to do this god
            labels[self._labels["continuous"].astype(float).between(cutoff[i],
                                                                    cutoff[i+1],
                                                                    inclusive="right")] = class_names[i]
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
        # TODO handle if index is a list of column names or indices
        if index is not None:
            if isinstance(index, int):
                if index < len(self.dataset.columns):
                    return self.dataset.columns[index]
            elif isinstance(index, str):
                if index in list(self.dataset.columns):
                    return index
        return None

    def to_dict(self):

        d = {}

        d["Arguments"] = self.stored_args
        d["Name"] = self.name
        d["Filepath"] = self.filepath
        #d["Curation"] = self.curate
        d["Label Column"] = self._label_col
        d["SMILES Column"] = self._smiles_col
        d["Label"] = self._label
        d["File Hash"] = self.file_hash

        return d


    @property
    def active_label(self):
        return self.active_label

    @active_label.setter
    def active_label(self, value):
        if value not in self._labels.keys():
            raise ValueError(f"Error setting active label: {value} not in {self._labels.keys()}")
        else:
            self.active_label = value

    def get_active_label(self):
        return self.active_label

    def add_label(self, name, label):
        label = np.array(label)
        assert len(label.shape) == 1
        assert label.shape[0] == self.dataset.shape[0]
        self._labels[name] = label

    def get_label(self, name=None):
        if name is None:
            return self._labels
        return self._labels[name]

    def iter_labels(self):
        for key, val in self._labels.items():
            yield key, val

    def get_children(self):
        return self._children

    def get_child_dataset(self, name):
        return self.dataset.loc[self._children[name]]

    def get_child_mask(self, name):
        return self._children[name]

    def add_child(self, name, indices):
        self._children[name] = indices
        self.__setattr__("name", indices)

    def iter_children(self):
        for key, val in self._children.items():
            yield key, val

    def remove_child(self, name):
        if name in self._children.keys():
            del self._children[name]
            delattr(self, name)

    def calc_descriptor(self, name):
        func_call = "calc_"+str(name)
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

    def model(self, model, name=None, child=None, desc="all", label=None, metrics=None, cv=None, save_models=True,
              verbose=False):
        #cv should be scikit splitter (like kfold)

        if hasattr(model, "random_state"):
            model.__setattr__("random_state", self._random_state)

        if name is None:
            name = len(self._models)
            while name in self._models.keys():
                name += 1

        if isinstance(desc, str):
            if desc == "all":
                desc = self.descriptor.get_all_descriptors()
            else:
                desc = [tuple([desc]) + self.descriptor.__getattribute__(desc)]
        else:
            desc = [tuple([d]) + self.descriptor.__getattribute__(d) for d in desc]

        if label is None:
            label = self.active_label

        y = self._labels[label]

        if child is None:
            child = self.get_children()
        elif isinstance(child, str):
            child = [self.get_child_mask(child)]
        else:
            child = [self.get_child_mask(c) for c in child]

        # TODO use itertools product to make this one loop so we can tqdm it
        for child_name, child_mask in child:
            for desc_name, desc_setting, X in desc:
                if cv is not None:
                    cv_stats = []
                    _m = QSARModel(self, model, name, child_mask, child_name, desc_name, desc_setting, label)
                    for i, train_index, val_index in enumerate(cv.split(X, y)):
                        X_train, X_val = X[train_index], X[val_index]
                        y_train, y_val = y[train_index], y[val_index]
                        m = copy.deepcopy(model)
                        _name = str(name) + f"_{i}"
                        m.fit(X_train, y_train)
                        m = QSARModel(self, m, name, child_mask, child_name, desc_name, desc_setting, label)
                        if save_models:
                            self._models[name] = m
                        self._screen(m, X_val, y_val, metircs)
                        if verbose:
                            self._print_screening_header(self, m.name, desc_name, child_name, label, metrics)
                            pprint.pprint(m.screening_stats[self])
                        cv_stats.append({key: val(y, y_pred) for key, val in metrics.items()})

                    collected_cv_stats = {}
                    for d in cv_stats:
                        for key in d:
                            if key in collected_cv_stats:
                                collected_cv_stats[key].append(d[key])
                            else:
                                collected_cv_stats[key] = [d[key]]
                                
                    # mean
                    cv_mean_stats = {key: np.average(np.array(val)) for key, val in collected_cv_stats.items()}
                    # std
                    cv_std_stats = {key: np.std(np.array(val)) for key, val in collected_cv_stats.items()}
                    
                    self._cv[_m] = {"mean": cv_mean_stats, "std": cv_std_stats}
                else:
                    m = copy.deepcopy(model)
                    m.fit(X, y)
                    m = QSARModel(self, m, name, child_mask, child_name, desc_name, desc_setting, label)
                    if save_models:
                        self._models[name] = m
                    if metrics is not None:
                        self._screen(m, X, y)
                        m.screening_stats[screening_dataset] = {key: val(y, y_pred) for key, val in metrics.items()}
                        if verbose:
                            self._print_screening_header(self, m.name, desc_name, child_name, label, metrics)
                            pprint.pprint(m.screening_stats[self])

    def _print_screening_header(self, model_name, screening_df, desc_name, child_name, label, metrics):
        print(f"Model {model_name}:\n"
              f"Trained on:\n"
              f"\t{self.name} with descriptor set {desc_name} with settings f{desc_setting}\n"
              f"\tchild mask {child_name}\n"
              f"\tlabel {label}"
              f"\n"
              f"Screened against {screening_df.name}\n"
              f"\n"
              f"Metrics use {list(metrics.keys())}")

    def get_models(self, name=None):
        if name is None:
            return self._models
        if isinstance(name, str):
            return self._models[name]
        return {n: self._models[n] for n in name}

    def remove_model(self, name):
        if name in self._models.keys():
            del self._models[name]

    def curate(self):

        from curate import curate_mol
        results = [curate_mol(x) for x in self.dataset["ROMol"]]

        passed = [x[1].passed for x in results]
        curated_mols = [x[0] for x in results]
        histories = [x[1] for x in results]
        modified = [x[1].structure_modified for x in results]

        failed = [not x[1].passed for x in results]
        new_fail_dict = {x:"Did not pass automatic curation" for x in self.dataset[failed].index}
        self._failed.update(new_fail_dict)

        self.dataset["Curation history"] = histories
        self.dataset["Passed curation"] = passed
        self.dataset["Curation modified structure"] = modified
        self.dataset["Curated ROMol"] = curated_mols
            
    def screen(self, screening_dataset, model=None, metrics=None):
        if model is None:
            model = list(self.get_models().values())
        elif not isinstance(model, (tuple, list)):
            if isinstance(model, str):
                model = [self.get_models(model)[model]]
            else:
                model = [model]
        else:
            if isinstance(model[0], QSARModel):
                model = model
            elif isinstance(model[0], str):
                model = [self.get_models(m)[m] for m in model]
        if not isinstance(model[0], QSARModel):
            raise ValueError(f"models are not valid. Make sure you are passing names of models that exist for this "
                             f"dataset object or QSARModel objects. view model names with .get_models()")

        if metrics is None:
            if getattr(model, "_estimator_type", None) == "classifier":
                metrics = "all_classification"
            elif getattr(model, "_estimator_type", None) == "regressor":
                metrics = "all_regression"
            else:
                raise ValueError("model does not define a _estimator_type, metrics must be stated explicitly")

        metric_functions = __import__(metrics)
        if isinstance(metrics, str):
            if metrics == "all_classification":
                metrics = getattr(metric_functions, "get_classification_metrics")()
            elif metrics == "all_regression":
                metrics = getattr(metric_functions, "get_regression_metrics")()
            else:
                metrics = {metrics: getattr(metric_functions, metrics)}
        else:
            metrics = {m: getattr(metric_functions, m) for m in metrics}

        for m in model:
            desc_name = m.desc_name
            if desc_name in screening_dataset.descriptor.__dict__.keys():
                desc = screening_dataset.descriptor.__getattribute__(desc_name)
                desc_settings = m.desc_settings
                if desc[0] != desc_settings:
                    screening_dataset.calc_descriptor(desc_name)
                    X = screening_dataset.descriptor.__getattribute__(desc_name)[1]
                else:
                    X = desc[1]
            else:
                # TODO add in support to auto detect and calculate native merged descriptors
                raise ValueError(f"Could not find matching descriptor set in screening dataset. Model descriptor set is"
                                 f" {desc_name} and screening dataset only has "
                                 f"{[_ for _ in screening_dataset.descriptor.__dict__.keys() if _ != '_cache']} "
                                 f"and can only calculate {screening_dataset.descriptor.get_descriptor_funcs}")

            label = m.label
            if label in screening_dataset.get_label().keys():
                y = screening_dataset.get_label(label)[label]
            else:
                raise ValueError(f"screening set label and model label do not match: model label is {label} screening "
                                 f"labels are {list(screening_dataset.get_label().keys())}")

            self._screen(m, X, y, metrics)
            m.screening_stats[screening_dataset] = {key: val(y, y_pred) for key, val in metrics.items()}

    @staticmethod
    def _screen(model, X, y):
        y_pred = model.model.pred(X, y)
        model.__setattr__("pred", y_pred)
        # lol I'm 70% confident that python classes hash to their mem loc id, so I can use it as a dict key

    def balance(self, method="downsample"):
        raise NotImplemented
        # TODO implement function
