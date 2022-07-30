import os
import yaml


class Plate:
    def __init__(self, datasets=None, models=None, descriptor_functions=None, sampling_methods=None, procedures=None):

        self.datasets = datasets if datasets is not None else []
        self.models = models if models is not None else []
        self.descriptor_functions = descriptor_functions if descriptor_functions is not None else []
        self.sampling_methods = sampling_methods if sampling_methods is not None else []
        self.procedures = procedures if procedures is not None else []  # cross-val, training, or screening

    def to_yaml(self, filename):
        from datetime import datetime

        host_dict = os.uname()
        s = {
            "Datasets": [x.to_dict() for x in self.datasets], "Models": [x.to_dict() for x in self.models],
            "Descriptor Functions": [x.to_dict() for x in self.descriptor_functions],
            "Sampling Methods": [x for x in self.sampling_methods], "Procedures": [x for x in self.procedures],
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
        from dataset import QSARDataset

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

