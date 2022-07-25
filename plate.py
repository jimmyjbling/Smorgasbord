import os
import yaml

def KFoldSplitter(x, y, n_folds = 5):

    from sklearn.model_selection import KFold
    kf = KFold(n_splits = n_folds)
    for train_indices, test_indices in kf.split(x):
        yield (x[train_indices], x[test_indices], y[train_indices], y[test_indices])


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
        import pandas as pd

        modeling_results = []
        for this_dataset, this_model, this_descriptor_function in itertools.product(self.datasets, self.models, self.descriptor_functions):

            s = pd.Series()
            s["Dataset"] = this_dataset.name
            s["Model"] = this_model.get_string()
            s["Descriptor"] = this_descriptor_function.get_string()


            descriptor_matrix = this_descriptor_function.get_descriptors(this_dataset.get_dataset()["ROMol"])
            if "binary" not in this_dataset._labels:
                this_dataset.to_binary()
            labels = this_dataset.get_labels(kind = "binary")

            from qsar_utils import modi
            s["MODI"] = modi(descriptor_matrix, labels)

            from sklearn.model_selection import train_test_split

            results = []

            for i, indx in enumerate(KFoldSplitter(descriptor_matrix, labels, n_folds = 5)):
                 
                s["Fold"] = i

                x_train, x_test, y_train, y_test = indx

                this_model.fit(x_train, y_train)
                pred = this_model.predict_probability(x_test)
                train_pred = this_model.predict_probability(x_train)

                print(y_test)
                print(pred)


                test_s = s.copy()
                train_s = s.copy()
                test_s["target"] = "Test" 
                train_s["target"] = "Train"
                test_s["predictions"] = pred
                train_s["predictions"] = train_pred

                train_s["true labels"] = y_train
                test_s["true labels"] = y_test

                '''
                from qsar_utils import apd_screening
                train_ad = apd_screening(y = x_train, training_data=x_train, threshold=None, norm_func=None)
                test_ad = apd_screening(y = x_test, training_data=x_train, threshold=None, norm_func=None)
                train_coverage = sum(train_ad) / len(train_ad)
                test_coverage = sum(test_ad) / len(test_ad)

                train_s["ad coverage"] = train_coverage 
                test_s["ad coverage"] = test_coverage 
                '''

                from metrics import get_classification_metrics, auc
                train_s["auc"] = auc(y_train, train_pred)
                test_s["auc"] = auc(y_test, pred)

                for standard_threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:

                    
                    this_test_s = test_s.copy()
                    this_test_s["Classification threshold"] = standard_threshold
                    thresholded = pred >= standard_threshold
                    stats = get_classification_metrics(y_test, thresholded)

                    for key, val in stats.items():
                        if key in this_test_s:
                            raise Exception("Overwriting a statistic?")
                        this_test_s[key] = val

                    this_train_s = train_s.copy()
                    this_train_s["Classification threshold"] = standard_threshold
                    thresholded = train_pred >= standard_threshold
                    stats = get_classification_metrics(y_train, thresholded)

                    for key, val in stats.items():
                        if key in this_train_s:
                            raise Exception("Overwriting a statistic?")
                        this_train_s[key] = val

                    modeling_results.append(this_train_s)
                    modeling_results.append(this_test_s)

            modeling_results = pd.DataFrame(modeling_results)
            modeling_results = modeling_results.sort_values(by = ["target", "Classification threshold", "Fold"])
            modeling_results.drop(columns = ["predictions", "true labels"]).to_csv("stats.csv")
            print(modeling_results)
            exit()


            test_df = modeling_results[modeling_results["target"] == "Test"]
            train_df = modeling_results[modeling_results["target"] == "Train"]

            threshold_plot_2(test_df["true labels"], test_df["predictions"], filename = "test.png")
            threshold_plot_2(train_df["true labels"], train_df["predictions"], filename = "train.png")
            result_df = pd.DataFrame(results, columns = ["Metric", "Fold"])
            print(result_df)
            exit()

            file_string = this_dataset.name + "_" + this_model.name + "_" + this_descriptor_function.name
            print(file_string)
            file_string  = clean_name(file_string)
            print(file_string)

            extra_data = f"Dataset: {this_dataset.name}\nDescriptor: {this_descriptor_function.name}\nModel: {this_model.name}"
            threshold_plot_2(y_test, pred, file_string + "_test_threshold_figure.png", extra_data + "\nPredict on: Test")
            threshold_plot_2(y_train, train_pred, file_string + "_train_threshold_figure.png", extra_data + "\nPredict on: Train")

def clean_name(name):

    name = name.replace(".", "_")
    return name
    

def threshold_plot(y_true, y_pred, filename, extra_data = None):

    import pandas as pd
    if type(y_true) == pd.Series:
        y_true = list(y_true)
        y_pred = list(y_pred)

    print(y_true)


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


def threshold_plot_2(y_true, y_pred, filename, extra_data = None):

    import pandas as pd
    if type(y_true) == pd.Series:
        y_true = list(y_true)
        y_pred = list(y_pred)

    print(y_true)


    from metrics import get_classification_metrics, auc, ppv, npv, accuracy, balanced_accuracy

    ppvs = []
    npvs = []
    accuracies = []
    balanced_accuracies = []
    nums_active = []

    import numpy as np

    thresholds = np.arange(0,1,0.01)

    all_ppvs = []
    all_npvs = []
    all_accuracies = []
    all_balanced_accuracies = []
    all_nums_active = []

    for i in range(len(y_pred)):

        ppvs = []
        npvs = []
        accuracies = []
        balanced_accuracies = []
        nums_active = []

        this_y_pred = y_pred[i]
        this_y_true = y_true[i]

        for threshold in thresholds:

            thresholded = this_y_pred > threshold

            ppvs.append(ppv(this_y_true, thresholded))
            npvs.append(npv(this_y_true, thresholded))
            accuracies.append(accuracy(this_y_true, thresholded))
            balanced_accuracies.append(balanced_accuracy(this_y_true, thresholded))
            nums_active.append(sum(thresholded))

        all_ppvs.append(ppvs)
        all_npvs.append(npvs)
        all_accuracies.append(accuracies)
        all_balanced_accuracies.append(balanced_accuracies)
        all_nums_active.append(nums_active)


    ppv_mean = np.mean(np.array(all_ppvs), axis = 0)
    ppv_std = np.std(np.array(all_ppvs), axis = 0)

    npv_mean = np.mean(np.array(all_npvs), axis = 0)
    npv_std = np.std(np.array(all_npvs), axis = 0)

    accuracy_mean = np.mean(np.array(all_accuracies), axis = 0)
    accuracy_std = np.std(np.array(all_accuracies), axis = 0)

    balanced_accuracy_mean = np.mean(np.array(all_balanced_accuracies), axis = 0)
    balanced_accuracy_std = np.std(np.array(all_balanced_accuracies), axis = 0)

    num_active_mean = np.mean(np.array(all_nums_active), axis = 0)
    num_active_std = np.std(np.array(all_nums_active), axis = 0)

    import matplotlib.pyplot as plt

    plt.figure()
    fig, ax2 = plt.subplots()
    ax1 = ax2.twinx()
    ax2.plot(thresholds, ppv_mean, label = "PPV")
    ax2.fill_between(thresholds, ppv_mean - ppv_std, ppv_mean + ppv_std, alpha = 0.3)
    ax2.plot(thresholds, npv_mean, label = "NPV")
    ax2.fill_between(thresholds, npv_mean - npv_std, npv_mean + npv_std, alpha = 0.3)
    ax2.plot(thresholds, accuracy_mean, label = "Accuracy")
    ax2.fill_between(thresholds, accuracy_mean - accuracy_std, accuracy_mean + accuracy_std, alpha = 0.3)
    ax2.plot(thresholds, balanced_accuracy_mean, label = "Balanced Accuracy")
    ax2.fill_between(thresholds, balanced_accuracy_mean - balanced_accuracy_std, balanced_accuracy_mean + balanced_accuracy_std, alpha = 0.3)
    ax1.plot(thresholds, num_active_mean, color = "red", linestyle = "--")
    print(num_active_mean)
    print(num_active_std)
    #ax2.fill_between(thresholds, num_active_mean - num_active_std, num_active_mean + num_active_std, alpha = 0.3, color = "red")
    ax2.fill_between(thresholds, num_active_mean - num_active_std, num_active_mean + num_active_std, alpha = 1, color = "red")
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