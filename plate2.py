import os
import yaml

from datetime import datetime
from itertools import product
from functools import partial

from dataset import QSARDataset, ScreeningDataset
from descriptor import DescriptorCalculator
from sampling import Sampler
from procedure import Procedure


class Plate:
    def __init__(self, datasets=None, models=None, descriptor_functions=None, sampling_methods=None, procedures=None,
                 metrics=None, save_models=False, generate_report=False, output_dir=None, random_state=None):

        self.overall_results = None
        self.metrics = metrics

        self._save_models = save_models
        self._generate_reports = generate_report
        self._output_dir = output_dir if output_dir is not None else os.path.join(os.getcwd(), str(datetime.now()))
        self._random_state = random_state if random_state is not None else 42  # lol it is the answer

        self.datasets = []
        self.models = []
        self.descriptor_functions = []
        self.sampling_methods = []
        self.procedures = []

        self.procedure = Procedure(metrics=self.metrics, report=self._generate_reports,
                                   output_dir=self._output_dir, random_state=self._random_state)

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
            if not all([dataset.descriptor.func_exists(func) for func in self.descriptor_functions]):
                # if it is not currently existing try to steal the func from another existing dataset
                for func in self.descriptor_functions:
                    if not dataset.descriptor.func_exists(func):
                        found_func = self._find_desc_func(func)
                        if found_func is not None:
                            dataset.descriptor.set_descriptor_function(found_func.__name__, found_func)
                        else:
                            raise ValueError(f"new dataset object missing descriptor function {func}")
            # sets datasets random state to match the plate
            if hasattr(dataset, "random_state"):
                dataset.__setattr__("random_state", self._random_state)
            self.datasets.append(dataset)
        else:
            raise ValueError(f"dataset must be of type {type(QSARDataset)} not {type(dataset)}")

    def add_model(self, model):
        if "fit" in dir(model) and callable(model.__getattribute__("fit")):
            if "predict" in dir(model) and callable(model.__getattribute__("predict")):
                # set the random state the plates if the model has a random state to set
                if hasattr(model, "random_state"):
                    model.__setattr__("random_state", self._random_state)
                self.models.append(model)
            else:
                raise ValueError(f"model object lacks a callable predict function")
        else:
            raise ValueError(f"model object lacks a callable fit function")

    def add_descriptor_function(self, descriptor_name, descriptor_func=None, **kwargs):
        if not all([d.descriptor.func_exists(descriptor_name) for d in self.datasets]):
            # check if you can find a matching descriptor set in any of the dataset and if so use that
            if descriptor_func is None:
                descriptor_func = self._find_desc_func(descriptor_name)

            if descriptor_func is not None:
                [d.descriptor.set_descriptor_function(descriptor_name, descriptor_func) for d in self.datasets]
            else:
                raise ValueError(f"Descriptor function {descriptor_name} does not exist")

        # this chunk of code is to get the full argument list of the descriptor function...
        #  it is not the most ideal way, but it works. The issue steams from the freedom to use the same descriptor
        #  with different params and since descriptor funcs are functions in a descriptor clac object it will just
        #  overwrite them.
        import inspect
        dummy_calc = self.datasets[0].descriptor if len(self.datasets) > 0 else DescriptorCalculator()
        signature = inspect.signature(dummy_calc.get_descriptor_func(descriptor_name))
        default_args = {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}
        kwargs = default_args.update(kwargs)

        self.descriptor_functions.append((descriptor_name, kwargs))

    def add_sampling_method(self, sampling_method, sampling_func=None):
        if all([d.sampler.func_exists(sampling_method) for d in self.datasets]):
            self.sampling_methods.append(sampling_method)
        else:
            # check if you can find a matching sampling function in any of the dataset and if so use that
            if sampling_func is None:
                sampling_func = self._find_sample_func(sampling_method)

            if sampling_func is not None:
                [d.sampler.add_sampling_func(sampling_method, sampling_func) for d in self.datasets]
            else:
                raise ValueError(f"Descriptor function {sampling_method} does not exist for "
                                 f"all currently loaded datasets")

    def add_procedures(self, procedure, **kwargs):
        if procedure in dir(self.procedure) and callable(self.procedure.__getattribute__(procedure)):
            # there has to be a better way to do this ask josh
            if procedure == "screen":
                if "screening_dataset" not in kwargs:
                    raise ValueError("screen procedure require an additional keyword arg for screening_dataset")
                elif not isinstance(kwargs["screening_dataset"], ScreeningDataset):
                    raise ValueError(f"screening_dataset must of class ScreeningDataset found "
                                     f"{type(kwargs['screening_dataset'])}")
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
        # TODO check that all procedures are callable

    def _check_models(self):
        raise NotImplementedError
        # TODO add in check for random state

    def _check_datasets(self):
        raise NotImplementedError
        # TODO add in check for random state

    def run(self, print_output=False):
        # make sure plate and procedure agree
        self.procedure.report = self._generate_reports

        from tqdm import tqdm
        # calculate descriptors for every dataset first
        for dataset, (desc_func, kwargs) in tqdm(product(self.datasets, self.descriptor_functions)):
            dataset.get_descriptor(desc_func, **kwargs)

        # calculate sampling masks for everyone next
        for dataset, samp_func in tqdm(product(self.datasets, self.sampling_methods)):
            dataset.balance(samp_func)

        # now run all the procedures
        combos = self._make_combos()

        overall_results = {}

        for dataset, model, desc_func, samp_func, proc in tqdm(combos):
            res = proc(moodel=model, dataset=dataset, descriptor_func=desc_func, sampling_func=samp_func)

            overall_results[(dataset, model, desc_func, samp_func, proc)] = res

            if self._save_models:
                for key in res.keys():
                    file_name = self._to_string(dataset, model, desc_func, samp_func, proc) + ".pkl"
                    file_loc = os.path.join(self._output_dir, "models", file_name)

                    # trys to use manual model saving, otherwise defaults to trying to pickle
                    if hasattr(key, "save"):
                        key.save(file_loc)
                    else:
                        import pickle
                        with open(file_loc, 'wb') as f:
                            pickle.dump(key, f)

        self.overall_results = overall_results

        if print_output:
            self.pretty_print()

    def to_yaml(self, filename):
        import inspect

        host_dict = os.uname()

        # my goal for this dictionary is to make it as unreadable as possible lol
        s = {
            "Datasets": [d.to_dict() for d in self.datasets],
            "Models": [{'Name': m.__name__(), 'args': m.get_params()} for m in self.models],
            "Descriptor Functions": [{'Name': d[0], 'args': d[1]} for d in self.descriptor_functions],
            "Sampling Methods": [s for s in self.sampling_methods],
            "Procedures": [{p.__name__: {k: v.default for k, v in inspect.signature(p).parameters.items() if v.default
                                         is not inspect.Parameter.empty}} for p in self.procedures],
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

        with open(filename, 'w') as f:
            f.write(yaml.dump(s))
            f.close()

    def from_yaml(self, filename):
        with open(filename, 'r') as f:
            d = yaml.load(f, Loader=yaml.Loader)
            f.close()

        dataset_dicts = d["Datasets"] if "Datasets" in d.keys() else []
        model_dicts = d["Models"] if "Models" in d.keys() else []
        procedure_dicts = d["Procedures"] if "Procedures" in d.keys() else []
        descriptor_dicts = d["Descriptor Functions"] if "Descriptor Functions" in d.keys() else []
        sampling_methods = d["Sampling Methods"] if "Sampling Methods" in d.keys() else []

        # I think that loading a plate from yaml should have it run right away without the ability to edit the plate...
        if len(dataset_dicts) == 0 or len(model_dicts) == 0 or len(procedure_dicts) == 0 or len(descriptor_dicts) == 0:
            raise ValueError(f"this smorgasbord plate is missing some essential dishes:"
                             f"{', '.join([f'no {_} passed' for _ in [dataset_dicts, model_dicts, procedure_dicts, descriptor_dicts] if len(_) == 0])}")

        for dataset_dict in dataset_dicts:
            args = dataset_dict["args"] if dataset_dict["args"] is not None else {}
            self.add_dataset(QSARDataset(**args))

        for model_dict in model_dicts:
            args = model_dict["args"] if model_dict["args"] is not None else {}
            self.add_model(self._get_model(**args))

        for descriptor_dict in descriptor_dicts:
            args = descriptor_dict["args"] if descriptor_dict["args"] is not None else {}
            self.add_descriptor_function(descriptor_dict["Name"], **args)

        for sampling_method in sampling_methods:
            self.add_sampling_method(sampling_method)

        for procedure_dict in procedure_dicts:
            if isinstance(procedure_dict, dict):
                proc_name = list(procedure_dict.keys())[0]
                proc_args = procedure_dict[proc_name] if procedure_dict[proc_name] is not None else {}
                self.add_procedures(proc_name, **proc_args)
            else:
                self.add_procedures(procedure_dict)

    @staticmethod
    def _get_model(model_name, **kwargs):
        import model
        if model_name in dir(model):
            return model.__dict__[model_name](**kwargs)
        else:
            raise ValueError(f"model class of name {model_name} does not exist in the model.py")

    @staticmethod
    def _to_string(dataset, model, desc_func, samp_func, proc):
        return "_".join([dataset.name, dataset.descriptor.get_string(desc_func), samp_func, model.__name__,  proc.__name__])

    def _check_for_results(self, overall_results):
        if overall_results is None:
            if overall_results in self.__dict__ and self.overall_results is not None:
                overall_results = self.overall_results
            else:
                raise ValueError("plate has no results")
        return overall_results

    def pretty_print(self, overall_results=None):

        print("-----------------"
              "|    RESULTS    |"
              "-----------------\n")

        overall_results = self._check_for_results(overall_results)

        for dataset, model, desc_func, samp_func, proc in overall_results.keys():
            header = self._to_string(dataset, model, desc_func, samp_func, proc)
            print(header)
            res = overall_results[(dataset, model, desc_func, samp_func, proc)]

            # skip procs that do not have metrics associated with them
            if None in res.keys() or None in res.values():  # skips only training
                continue
            if isinstance(proc, partial) and "screening_dataset" in proc.keywords:  # skips screening
                continue

            for fold, (model, res) in enumerate(res.items()):
                print(f"FOLD {fold}:  {' | '.join([f'{str(key)}: {str(val)}' for key, val in res.items()])}")
            print("\n-------------------------------------------------------------------\n")

    def results_to_csv(self, overall_results=None):
        raise NotImplementedError
        # overall_results = self._check_for_results(overall_results)

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
        self.procedure.report = value

    @property
    def output_dir(self):
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value):
        self._output_dir = value
        self.procedure.output_dir = self._output_dir

    @output_dir.deleter
    def output_dir(self):
        self._output_dir = os.path.join(os.getcwd(), str(datetime.now()))
        self.procedure.output_dir = self._output_dir
