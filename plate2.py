import os
import yaml

from datetime import datetime
from itertools import product

from dataset import QSARDataset
from descriptor import DescriptorCalculator
from sampling import Sampler
from procedure import Procedure


class Plate:
    def __init__(self, datasets=None, models=None, descriptor_functions=None, sampling_methods=None, procedures=None,
                 metrics=None, save_models=False, generate_report=False, output_dir=None):

        self.metrics = metrics

        self._save_models = save_models
        self._generate_reports = generate_report
        self._output_dir = output_dir if output_dir is not None else os.path.join(os.getcwd(), str(datetime.now()))

        self.datasets = []
        self.models = []
        self.descriptor_functions = []
        self.sampling_methods = []
        self.procedures = []

        self.procedure = Procedure(metrics=self.metrics)

        if datasets is not None: [self.add_dataset(dataset) for dataset in self._check_list(datasets)]
        if models is not None: [self.add_model(model) for model in self._check_list(models)]
        if descriptor_functions is not None: [self.add_descriptor_function(desc_func) for desc_func in self._check_list(descriptor_functions)]
        if sampling_methods is not None: [self.add_sampling_method(samp_meth) for samp_meth in self._check_list(sampling_methods)]
        if procedures is not None: [self.add_procedures(procedure) for procedure in self._check_list(procedures)]

    @staticmethod
    def _check_list(arg):
        if not isinstance(arg, list):
            try:
                if isinstance(arg, str):
                    arg = [arg]
                else:
                    arg = list(arg)
            except TypeError:
                raise ValueError("parameters must be single objects or iterables of objects")
        return arg

    def _find_desc_func(self, name):
        func = None
        for d in self.datasets:
            if d.descriptor.func_exists(name):
                func = d.descriptor.get_descriptor_func(name)
                break
        return func

    def _find_sample_func(self, name):
        func = None
        for d in self.datasets:
            if d.sampler.func_exists(name):
                func = d.sampler.get_sampling_func(name)
                break
        return func

    def add_dataset(self, dataset):
        if isinstance(dataset, QSARDataset):
            # I added a check to see if a new dataset was compatible with the current descriptor sets
            if all([dataset.descriptor.func_exists(func) for func in self.descriptor_functions]):
                self.datasets.append(dataset)
            else:
                # if it is not currently existing try to steal the func from another existing dataset
                for func in self.descriptor_functions:
                    if not dataset.descriptor.func_exists(func):
                        found_func = self._find_desc_func(func)
                        if found_func is not None:
                            dataset.descriptor.add_custom_descriptor(found_func)
                        else:
                            raise ValueError(f"new dataset object missing descriptor function {func}")
        else:
            raise ValueError(f"dataset must be of type {type(QSARDataset)} not {type(dataset)}")

    def add_model(self, model):
        if "fit" in dir(model) and callable(model.__getattribute__("fit")):
            if "predict" in dir(model) and callable(model.__getattribute__("predict")):
                self.models.append(model)
            else:
                raise ValueError(f"model object lacks a callable predict function")
        else:
            raise ValueError(f"model object lacks a callable fit function")

    def add_descriptor_function(self, descriptor_name, descriptor_func=None):
        if all([d.descriptor.func_exists(descriptor_name) for d in self.datasets]):
            self.descriptor_functions.append(descriptor_name)
        else:
            # check if you can find a matching descriptor set in any of the dataset and if so use that
            if descriptor_func is None:
                descriptor_func = self._find_desc_func(descriptor_name)

            if descriptor_func is not None:
                [d.descriptor.add_custom_descriptor(descriptor_name, descriptor_func) for d in self.datasets]
            else:
                raise ValueError(f"Descriptor function {descriptor_name} does not exist for "
                                 f"all currently loaded datasets")

    def add_sampling_method(self, sampling_method, sampling_func=None):
        if all([d.sampler.func_exists(sampling_method) for d in self.datasets]):
            self.sampling_methods.append(sampling_method)
        else:
            # check if you can find a matching descriptor set in any of the dataset and if so use that
            if sampling_func is None:
                sampling_func = self._find_sample_func(sampling_method)

            if sampling_func is not None:
                [d.sampler.add_custom_sampling_func(sampling_method, sampling_func) for d in self.datasets]
            else:
                raise ValueError(f"Descriptor function {sampling_method} does not exist for "
                                 f"all currently loaded datasets")

    def add_procedures(self, procedure, **kwargs):
        from functools import partial
        if procedure in dir(self.procedure) and callable(self.procedure.__getattribute__(procedure)):
            self.procedures.append(partial(self.procedure.__getattribute__(procedure), **kwargs))
        else:
            raise ValueError(f"Procedure {procedure} does not exist")

    def add_metric(self):
        raise NotImplementedError

    def set_dataset_label(self, label_class):
        if all([label_class in d.get_existing_labels() for d in self.datasets]):
            for d in self.datasets:
                d.set_active_label(label_class)
        else:
            raise ValueError(f"Datasets on plate do not all have label of type {label_class}")

    def generate_binary_labels(self, cutoff, class_names=None):
        for d in self.datasets:
            d.to_binary(cutoff, class_names)

    def _make_combos(self):
        datasets = self.datasets if len(self.datasets) > 0 else [None]
        models = self.models if len(self.models) > 0 else [None]
        descriptor_functions = self.descriptor_functions if len(self.descriptor_functions) > 0 else [None]
        sampling_methods = self.sampling_methods if len(self.sampling_methods) > 0 else [None]
        procedures = self.procedures if len(self.procedures) > 0 else [None]

        return product(datasets, models, descriptor_functions, sampling_methods, procedures)

    def _check_metrics(self):
        if not all([callable(m) for m in self.metrics]):
            raise ValueError("Not all metrics are callable functions")

    def _check_labels(self):
        if len(set([d.get_label().dtype for d in self.datasets])) > 1:
            raise ValueError("Some labels are continuous and some are discrete")

    def _check_procedures(self):
        raise NotImplementedError

    def run(self):
        from tqdm import tqdm
        # calculate descriptors for every dataset first
        for dataset, desc_func in tqdm(product(self.datasets, self.descriptor_functions)):
            dataset.calc_descriptor(desc_func)

        # calculate sampling masks for everyone next
        for dataset, samp_func in tqdm(product(self.datasets, self.sampling_methods)):
            dataset.balance(samp_func)

        # now run all the procedures
        combos = self._make_combos()

        for dataset, model, desc_func, samp_func, proc in tqdm(combos):
            res = proc(moodel=model, dataset=dataset, descriptor_func=desc_func, sampling_func=samp_func)

    def to_yaml(self, filename):
        host_dict = os.uname()
        s = {
            "Datasets": [x.to_dict() for x in self.datasets],
            "Models": [x.to_dict() for x in self.models],
            "Descriptor Functions": [x.to_dict() for x in self.descriptor_functions],
            "Sampling Methods": [x for x in self.sampling_methods],
            "Procedures": [x for x in self.procedures],
            "Metadata": {
                "Date": str(datetime.now()),
                "User": os.getlogin(),
                "Host": {
                    "OS Name": host_dict.sysname if "sysname" in host_dict.__dict__.keys() else None,
                    "Hostname": host_dict.nodename if "nodename" in host_dict.__dict__.keys() else None,
                    "Arch": host_dict.machine if "machine" in host_dict.__dict__.keys() else None
                }
            }
        }

        s = yaml.dump(s)

        with open(filename, 'w') as f:
            f.write(s)
            f.close()

    def from_yaml(self, filename, check_file_contents=True):
        with open(filename, 'r') as f:
            d = yaml.load(f, Loader=yaml.Loader)
            f.close()

        # print(d)  #DEBUG LINE
        dataset_dicts = d["Datasets"]

        for dataset_dict in dataset_dicts:
            # print(dataset_dict)  #DEBUG LINE
            args = dataset_dict["Arguments"]
            if check_file_contents and "File Hash" in dataset_dict:
                self.datasets.append(QSARDataset(file_hash=dataset_dict["File Hash"], **args))
            else:
                self.datasets.append(QSARDataset(**args))

    @property
    def save_models(self):
        return self._save_models

    @save_models.setter
    def save_models(self, value):
        self._save_models = value

    @property
    def generate_reports(self):
        return self._generate_reports

    @generate_reports.setter
    def generate_reports(self, value):
        self._generate_reports = value

    @property
    def output_dir(self):
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value):
        self._output_dir = value

    @output_dir.deleter
    def output_dir(self):
        self._output_dir = os.path.join(os.getcwd(), str(datetime.now()))