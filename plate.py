import os
import yaml


class Plate:

    def __init__(self, datasets = [], models = [], descriptor_functions = [], sampling_methods = [], procedures = []):

        self.datasets = datasets
        self.models = models
        self.descriptor_functions = descriptor_functions
        self.sampling_methods = sampling_methods
        self.procedures = procedures #cross-val, training, or screening

    def to_yaml(self, filename):

        '''
        def represent_none(self, _):
            return self.represent_scalar('tag:yaml.org,2002:null', 'None')

        yaml.add_representer(type(None), represent_none)
        '''

        s = {}

        s["Datasets"] = [x.to_dict() for x in self.datasets]
        s["Models"] = [x.to_dict() for x in self.models]
        s["Descriptor Functions"] = [x.to_dict() for x in self.descriptor_functions]
        s["Sampling Methods"] = [x for x in self.sampling_methods]
        s["Procedures"] = [x for x in self.procedures]

        from datetime import datetime
        s["Metadata"] = {}
        s["Metadata"]["Date"] = str(datetime.now())
        s["Metadata"]["User"] = os.getlogin()

        try:
            host_dict = os.uname()
            d = {}
            d["OS Name"] = host_dict[0]
            d["Hostname"] = host_dict[1]

            #maybe a security problem lol
            #d["Kernel"] = host_dict[2]
            #d["Flavor"] = host_dict[3]
            d["Arch"] = host_dict[4]
            s["Metadata"]["Host"] = d
        except:
            pass

        s = yaml.dump(s)

        
        f = open(filename, 'w')
        f.write(s)
        f.close()
            
    @staticmethod
    def from_yaml(filename, check_file_contents = True):

        from yaml import Loader
        f = open(filename, 'r')
        d = yaml.load(f, Loader = Loader)
        f.close()
        print(d)

        from dataset import QSARDataset
        dataset_dicts = d["Datasets"]
        for dataset_dict in dataset_dicts:
            print(dataset_dict)
            args = dataset_dict["Arguments"]
            if check_file_contents:
                file_hash = None
                if "File Hash" in dataset_dict:
                    file_hash = dataset_dict["File Hash"]

                dataset = QSARDataset(file_hash = file_hash, **args)
            else:
                dataset = QSARDataset(**args)

    def run(self):

        import itertools
        for this_dataset, this_model, this_descriptor_function in itertools.product(self.datasets, self.models, self.descriptor_functions):

            descriptor_matrix = this_descriptor_function.get_descriptors(this_dataset.get_dataset()["ROMol"])
            if "binary" not in this_dataset._labels:
                this_dataset.to_binary()
            labels = this_dataset.get_labels(kind = "binary")


            from sklearn.model_selection import train_test_split


            x_train, x_test, y_train, y_test = train_test_split(descriptor_matrix, labels, test_size = 0.5)

            this_model.fit(x_train, y_train)
            pred = this_model.predict_probability(x_test)
            train_pred = this_model.predict_probability(x_train)

            print(y_test)
            print(pred)



            from metrics import get_classification_metrics, auc
            #stats = get_classification_metrics(y_test, pred)
            auc = auc(y_test, pred)
            print(auc)

            file_string = this_dataset.name + "_" + this_model.name + "_" + this_descriptor_function.name
            print(file_string)
            file_string  = clean_name(file_string)
            print(file_string)

            extra_data = f"Dataset: {this_dataset.name}\nDescriptor: {this_descriptor_function.name}\nModel: {this_model.name}"
            threshold_plot(y_test, pred, file_string + "_test_threshold_figure.png", extra_data + "\nPredict on: Test")
            threshold_plot(y_train, train_pred, file_string + "_train_threshold_figure.png", extra_data + "\nPredict on: Train")

def clean_name(name):

    name = name.replace(".", "_")
    return name
    

def threshold_plot(y_true, y_pred, filename, extra_data = None):

    from metrics import get_classification_metrics, auc, ppv, npv, accuracy, balanced_accuracy

    ppvs = []
    npvs = []
    accuracies = []
    balanced_accuracies = []
    nums_active = []
    import numpy as np

    #make super stat plot
    thresholds = np.arange(0,1,0.01)
    for threshold in thresholds:

        thresholded = y_pred > threshold

        ppvs.append(ppv(y_true, thresholded))
        npvs.append(npv(y_true, thresholded))
        accuracies.append(accuracy(y_true, thresholded))
        balanced_accuracies.append(balanced_accuracy(y_true, thresholded))
        nums_active.append(sum(thresholded))


    import matplotlib.pyplot as plt
    plt.figure()
    fig, ax2 = plt.subplots()
    ax1 = ax2.twinx()
    ax2.plot(thresholds, ppvs, label = "PPV")
    ax2.plot(thresholds, npvs, label = "NPV")
    ax2.plot(thresholds, accuracies, label = "Accuracy")
    ax2.plot(thresholds, balanced_accuracies, label = "Balanced Accuracy")
    ax1.plot(thresholds, nums_active, color = "red", linestyle = "--")
    ax2.yaxis.label.set_text("Value of Metric")
    ax1.yaxis.label.set_text("--- Number of predicted actives")
    ax1.yaxis.label.set_color("red")
    ax2.set_xlabel("Classification Threshold")
    ax1.spines['right'].set_color('red')
    ax1.tick_params(axis = 'y', colors = 'red')
    plt.title("Validation Stats as a Function of Changing Threshold")
    ax2.set_xticks(np.arange(0,1.1,0.1))
    ax2.set_yticks(np.arange(0,1.1,0.1))
    ax2.set_xlim((0, 1))
    ax2.set_ylim((0, 1))
    ax1.set_xlim((0, 1))
    ax1.set_ylim((0, max(nums_active)))
    ax2.grid(visible = True, axis = 'both', alpha = 0.5)
    ax2.legend(bbox_to_anchor=(-0.1, 1), loc='upper right', ncol=1)

    if extra_data:
        ax2.text(-0.5, 0.5, extra_data, fontsize = 8)
    plt.savefig(filename, bbox_inches = "tight", dpi = 300)
