import os
import shutil
import yaml


class KFoldSplitter:

    def __init__(self, n_folds = 5):
        self.n_folds = n_folds

    def split(self, x):
        from sklearn.model_selection import KFold
        kf = KFold(n_splits = self.n_folds, shuffle = True)
        for train_indices, test_indices in kf.split(x):
            yield (train_indices, test_indices)

    def get_name(self):

        return f"{self.n_folds}-Fold Cross Validation"


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

    def run(self, output_directory = "default_output_directory", overwrite = False):

        if output_directory == "default_output_directory":
            print(f"WARNING: using {output_directory} for output...")
        else:
            print(f"Using {output_directory} for output...")

        if os.path.exists(output_directory):
            if overwrite:
                shutil.rmtree(output_directory)
            else:
                raise Exception(f"Will not overwrite existing directory {output_directory} without setting 'overwrite' to True")

        os.makedirs(output_directory)

        import itertools
        import pandas as pd

        ###OVERRIDE
        ###OVERRIDE
        ###OVERRIDE
        self.splitters = [KFoldSplitter(n_folds = 5)]
        ###OVERRIDE
        ###OVERRIDE
        ###OVERRIDE
        modeling_results = []
        for this_dataset in self.datasets:
            for this_splitter in self.splitters:

                this_folds = [x for x in this_splitter.split(this_dataset.get_dataset())]

                for this_model, this_descriptor_function in itertools.product(self.models, self.descriptor_functions):

                    s = pd.Series(dtype = object)
                    s["Dataset"] = this_dataset.name
                    s["Model"] = this_model.get_string()
                    s["Descriptor"] = this_descriptor_function.get_string()
                    s["Splitter"] = this_splitter.get_name()


                    print(f"Running {this_dataset.name}, {this_descriptor_function.name}, {this_model.name}...")
                    descriptor_matrix = this_descriptor_function.get_descriptors(this_dataset.get_dataset()["ROMol"])

                    s["Dataset Size"] = len(descriptor_matrix)

                    from normalize import Normalizer

                    
                    for this_task in this_dataset.tasks:
                        if this_task == "classification":

                            s["Task"] = "classification"

                            #descriptor_matrix = descriptor_matrix > 0
                            if "binary" not in this_dataset._labels:
                                this_dataset.to_binary()
                            labels = this_dataset.get_labels(kind = "binary")

                            inchis = [Chem.MolToInchi(x) for x in this_dataset.get_dataset()["ROMol"]]

                            normalizer = Normalizer()

                            normalizer.fit(descriptor_matrix, labels)
                            descriptor_matrix = normalizer.transform(descriptor_matrix)

                            '''
                            from qsar_utils import modi
                            print("Calculating MODI...")
                            s["MODI"] = modi(descriptor_matrix, labels)
                            '''

                            from sklearn.model_selection import train_test_split

                            results = []

                            fold_ids = {}

                            for i, indx in enumerate(this_folds):
                                print(f"Fold: {i}")
                                 
                                s["Fold"] = i

                                x_train = descriptor_matrix[train_indices]
                                y_train = labels[train_indices]

                                x_test = descriptor_matrix[test_indices]
                                y_test = labels[test_indices]


                                s["Train Size"] = len(x_train)
                                s["Test Size"] = len(x_test)

                                train_indices, test_indices = indx
                                '''
                                from model import NN
                                model = NN()
                                model.fit(x_train, y_train)
                                exit()
                                '''

                                #this_model.reset()
                                this_model.fit(x_train, y_train, output_filename = f"{output_directory}/loss_test.png")
                                pred = this_model.predict_probability(x_test)
                                train_pred = this_model.predict_probability(x_train)

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

                                    results.append(this_train_s)
                                    results.append(this_test_s)

                                    modeling_results.append(this_train_s)
                                    modeling_results.append(this_test_s)


                            
                            result_df = pd.DataFrame(results)
                            test_df = result_df[(result_df["target"] == "Test") & (result_df["Classification threshold"] == 0.5)]
                            train_df = result_df[(result_df["target"] == "Train") & (result_df["Classification threshold"] == 0.5)]
                            extra_data = f"Dataset: {this_dataset.name}\nDescriptor: {this_descriptor_function.name}\nModel: {this_model.name}\n"
                            threshold_plot_2(test_df["true labels"], test_df["predictions"], filename = f"{output_directory}/{this_model.name}_{this_descriptor_function.name}_test.png", extra_data = extra_data)
                            threshold_plot_2(train_df["true labels"], train_df["predictions"], filename = f"{output_directory}/{this_model.name}_{this_descriptor_function.name}_train.png", extra_data = extra_data)


                        elif this_task == "regression":

                            if "continuous" not in this_dataset._labels:
                                raise Exception("come on dude")

                            s["Task"] = "regression"

                            labels = this_dataset.get_labels(kind = "continuous")

                            normalizer = Normalizer()

                            normalizer.fit(descriptor_matrix, labels)
                            descriptor_matrix = normalizer.transform(descriptor_matrix)

                            from sklearn.model_selection import train_test_split

                            results = []

                            for i, indx in enumerate(this_folds):
                                print(f"Fold: {i}")
                                 
                                s["Fold"] = i

                                train_indices, test_indices = indx

                                x_train = descriptor_matrix[train_indices]
                                y_train = labels[train_indices]

                                x_test = descriptor_matrix[test_indices]
                                y_test = labels[test_indices]


                                s["Train Size"] = len(x_train)
                                s["Test Size"] = len(x_test)

                                #this_model.reset()
                                if this_model.name == "NNRegressor":
                                    this_model.fit(x_train, y_train, output_filename = f"{output_directory}/nnregressor_loss_fold_{i}.png")
                                else:
                                    this_model.fit(x_train, y_train)
                                pred = this_model.predict(x_test)
                                train_pred = this_model.predict(x_train)

                                test_s = s.copy()
                                train_s = s.copy()
                                test_s["target"] = "Test" 
                                train_s["target"] = "Train"
                                test_s["predictions"] = pred
                                train_s["predictions"] = train_pred

                                train_s["true labels"] = y_train
                                test_s["true labels"] = y_test

                                #TODO: put r2 into metrics
                                from sklearn.metrics import r2_score as r2
                                from sklearn.metrics import mean_absolute_error as mae
                                from sklearn.metrics import mean_squared_error as mse

                                train_s["r2"] = r2(y_train, train_pred)
                                test_s["r2"] = r2(y_test, pred)

                                train_s["mae"] = mae(y_train, train_pred)
                                test_s["mae"] = mae(y_test, pred)

                                train_s["mse"] = mse(y_train, train_pred)
                                test_s["mse"] = mse(y_test, pred)

                                results.append(train_s)
                                results.append(test_s)

                                modeling_results.append(train_s)
                                modeling_results.append(test_s)

                            
                            result_df = pd.DataFrame(results)
                            print(result_df)


                            #plot scatter

                            train_preds = result_df["predictions"][result_df["target"] == "Train"]
                            train_labels = result_df["true labels"][result_df["target"] == "Train"]
                            test_preds = result_df["predictions"][result_df["target"] == "Test"]
                            test_labels = result_df["true labels"][result_df["target"] == "Test"]

                            import matplotlib.pyplot as plt
                            plt.figure()
                            for i in range(len(train_preds)):
                                    plt.scatter(test_labels.iloc[i], test_preds.iloc[i], label = f"Fold {i}", alpha = 0.5)

                            plt.legend()
                            plt.xlabel("Ground Truth")
                            plt.ylabel("Prediction")
                            plt.savefig(f"{output_directory}/{this_model.name}_{this_descriptor_function.name}_scatter_test.png")

                            import matplotlib.pyplot as plt
                            plt.figure()
                            for i in range(len(train_preds)):
                                    plt.scatter(train_labels.iloc[i], train_preds.iloc[i], label = f"Fold {i}", alpha = 0.5)

                            plt.legend()
                            plt.xlabel("Ground Truth")
                            plt.ylabel("Prediction")
                            plt.savefig(f"{output_directory}/{this_model.name}_{this_descriptor_function.name}_scatter_train.png")


                        else:
                            raise Exception("come on dude")

        modeling_results = pd.DataFrame(modeling_results)

        classification_results = modeling_results[modeling_results["Task"] == "classification"]
        regression_results = modeling_results[modeling_results["Task"] == "regression"]

        if len(classification_results) > 0:
            classification_results = classification_results.sort_values(by = ["target", "Classification threshold", "Fold"])
            classification_results.drop(columns = ["predictions", "true labels"]).to_csv(f"{output_directory}/classification_validation_statistics.csv")

        if len(regression_results) > 0:
            regression_results.drop(columns = ["predictions", "true labels"]).to_csv(f"{output_directory}/regression_validation_statistics.csv")

        for task, this_task_df in [("classification", classification_results), ("regression", regression_results)]:
            
            if task == "classification":
                if len(classification_results) <= 0:
                    continue

                final_datasets = this_task_df["Dataset"].unique()
                final_descriptors = this_task_df["Descriptor"].unique()
                final_models = this_task_df["Model"].unique()
                final_thresholds = this_task_df["Classification threshold"].unique()

                clean_results = []
                for this_dataset, this_model, this_descriptor_function, this_threshold in itertools.product(final_datasets, final_models, final_descriptors, final_thresholds):
                    subdf = this_task_df[(this_task_df["Dataset"] == this_dataset) &
                                             (this_task_df["Descriptor"] == this_descriptor_function) & 
                                             (this_task_df["Model"] == this_model) &
                                             (this_task_df["Classification threshold"] == this_threshold)]

                    if len(subdf) == 0:
                        continue

                    for target in ["Train", "Test"]:
                        this_subdf = subdf[subdf["target"] == target]

                        import numpy as np

                        ppv_mean = np.mean(np.array(this_subdf["ppv"]), axis = 0)
                        ppv_std = np.std(np.array(this_subdf["ppv"]), axis = 0)

                        npv_mean = np.mean(np.array(this_subdf["npv"]), axis = 0)
                        npv_std = np.std(np.array(this_subdf["npv"]), axis = 0)

                        accuracy_mean = np.mean(np.array(this_subdf["accuracy"]), axis = 0)
                        accuracy_std = np.std(np.array(this_subdf["accuracy"]), axis = 0)

                        balanced_accuracy_mean = np.mean(np.array(this_subdf["balanced_accuracy"]), axis = 0)
                        balanced_accuracy_std = np.std(np.array(this_subdf["balanced_accuracy"]), axis = 0)

                        print(ppv_mean)
                        print(ppv_std)

                        s = pd.Series(dtype = object)

                        s["Dataset"] = this_dataset
                        s["Descriptor"] = this_descriptor_function
                        s["Model"] = this_model
                        s["Target"] = target
                        s["Threshold"] = this_threshold

                        s["PPV (mean)"] = ppv_mean
                        s["PPV (std dev)"] = ppv_std

                        s["NPV (mean)"] = npv_mean
                        s["NPV (std dev)"] = npv_std

                        s["Accuracy (mean)"] = accuracy_mean
                        s["Accuracy (std dev)"] = accuracy_std

                        s["Balanced Accuracy (mean)"] = balanced_accuracy_mean
                        s["Balanced Accuracy (std dev)"] = balanced_accuracy_std

                        clean_results.append(s)

                clean_df = pd.DataFrame(clean_results)
                clean_df = clean_df.sort_values(by = ["Target", "Dataset", "Descriptor", "Model"])
                clean_df.to_csv(f"{output_directory}/clean_classification_validation_stats.csv")
            elif task == "regression":

                final_datasets = this_task_df["Dataset"].unique()
                final_descriptors = this_task_df["Descriptor"].unique()
                final_models = this_task_df["Model"].unique()

                clean_results = []
                for this_dataset, this_model, this_descriptor_function in itertools.product(final_datasets, final_models, final_descriptors):
                    subdf = this_task_df[(this_task_df["Dataset"] == this_dataset) &
                                             (this_task_df["Descriptor"] == this_descriptor_function) & 
                                             (this_task_df["Model"] == this_model)]

                    if len(subdf) == 0:
                        continue

                    for target in ["Train", "Test"]:
                        this_subdf = subdf[subdf["target"] == target]

                        import numpy as np

                        r2_mean = np.mean(np.array(this_subdf["r2"]), axis = 0)
                        r2_std = np.std(np.array(this_subdf["r2"]), axis = 0)

                        mae_mean = np.mean(np.array(this_subdf["mae"]), axis = 0)
                        mae_std = np.std(np.array(this_subdf["mae"]), axis = 0)

                        mse_mean = np.mean(np.array(this_subdf["mse"]), axis = 0)
                        mse_std = np.std(np.array(this_subdf["mse"]), axis = 0)

                        s = pd.Series(dtype = object)

                        s["Dataset"] = this_dataset
                        s["Descriptor"] = this_descriptor_function
                        s["Model"] = this_model
                        s["Target"] = target

                        s["R2 (mean)"] = r2_mean
                        s["R2 (std dev)"] = r2_std

                        s["MAE (mean)"] = mae_mean
                        s["MAE (std dev)"] = mae_std

                        s["MSE (mean)"] = mse_mean
                        s["MSE (std dev)"] = mse_std

                        clean_results.append(s)

                clean_df = pd.DataFrame(clean_results)
                clean_df = clean_df.sort_values(by = ["Target", "Dataset", "Descriptor", "Model"])
                clean_df.to_csv(f"{output_directory}/clean_regression_validation_stats.csv")



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
    plt.close()
