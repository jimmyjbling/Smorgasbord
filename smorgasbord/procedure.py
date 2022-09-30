from metrics import get_default_classification_metrics, get_default_regression_metrics


class Procedure:
    def __init__(self, metrics=None, report=False, output_dir=None, random_state=None):
        self._random_state = random_state
        if self._random_state is not None:
            self._shuffle = True
        else:
            self._shuffle = False
        self.report = report
        self.metrics = metrics
        self.output_dir = output_dir

    # this function takes in kwargs because the way I make the plates run it will always pass dataset and samp func
    #  kwargs will allow this to work without an argument error (rusty and bad practice sure, but it works)
    def screen(self, model, descriptor_func, screening_dataset, **kwargs):
        screening_X = screening_dataset.get_descriptor(descriptor_func)

        if "predict_proba" in dir(model) and callable(model.__getattribute__("predict_proba")):
            res = model.predict_proba(screening_X)
        else:
            res = model.predict(screening_X)

        return {screening_dataset.name: res}

    def train(self, model, dataset, descriptor_func, sampling_func):

        y = dataset.get_label(mask_name=sampling_func)
        X = dataset.get_descriptor(descriptor_func, mask_name=sampling_func)

        model.fit(X, y)

        return {model: None}

    def train_and_screen(self, dataset, sampling_func, model, descriptor_func, screening_dataset, **kwargs):
        self.train(model=model, dataset=dataset, descriptor_func=descriptor_func, sampling_func=sampling_func)
        return self.screen(model=model, descriptor_func=descriptor_func, screening_dataset=screening_dataset)

    def train_with_test(self, model, dataset, descriptor_func, sampling_func, cv=None, **kwargs):

        if cv is None:
            from sklearn.model_selection import StratifiedShuffleSplit
            cv = StratifiedShuffleSplit
            kwargs["n_splits"] = 1
            kwargs["test_size"] = 0.2

        kwargs["random_state"] = self._random_state

        s = cv(**kwargs)

        y = dataset.get_label(mask_name=sampling_func)
        X = dataset.get_descriptor(descriptor_func, mask_name=sampling_func)

        train_index, test_index = next(s.split(X, y))

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)

        if "predict_proba" in dir(model) and callable(model.__getattribute__("predict_proba")):
            y_pred = model.predict_proba(X_test)
        else:
            y_pred = model.predict(X_test)

        res = self._eval(y_test, y_pred)

        # TODO implement the report functions
        if self.report:
            pass

        return {model: (y_test, y_pred, res)}

    def cross_validate(self, model, dataset, descriptor_func, sampling_func, cv=None, **kwargs):
        from copy import deepcopy
        if cv is None:
            from sklearn.model_selection import StratifiedKFold, KFold
            if model.is_classifier():
                cv = StratifiedKFold
            else:
                cv = KFold
        else:
            from sklearn import model_selection
            if cv in dir(model_selection):
                cv = model_selection.__dict__[cv]

        kwargs["random_state"] = self._random_state
        kwargs["shuffle"] = self._shuffle

        s = cv(**kwargs)

        cv_models = {}

        # Use None sampling here for the whole dataset to not skew the test set
        y = dataset.get_label(mask_name=None)
        X = dataset.get_descriptor(descriptor_func, mask_name=None)

        import numpy as np
        for train_index, test_index in s.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Do sampling on JUST training data
            if sampling_func is not None:
                X_train, y_train = dataset.sampler.get_func(mask_name=sampling_func)(X_train, y_train)

            print('Sampling func:', sampling_func)
            print('Dataset sizes:', y_train.shape, y_test.shape)
            print('Train balance:', np.unique(y_train, return_counts=True))
            print('Test balance:', np.unique(y_test, return_counts=True))

            model_copy = deepcopy(model)

            model_copy.fit(X_train, y_train)

            # TODO this needs to be rethought and fixed
            # if "predict_proba" in dir(model_copy) and callable(model_copy.__getattribute__("predict_proba")):
            #     y_pred = model_copy.predict_proba(X_test)
            # else:
            #     y_pred = model_copy.predict(X_test)

            # tmp fix
            y_pred = model_copy.predict(X_test)

            res = self._eval(y_test, y_pred)

            # TODO implement report function
            if self.report:
                pass

            cv_models[model_copy] = (y_test, y_pred, res)

        return cv_models

    def _eval(self, y_true, y_pred):
        if self.metrics is None:
            if y_true.dtype == int:
                self.metrics = get_default_classification_metrics()
            else:
                self.metrics = get_default_regression_metrics()

        if not isinstance(self.metrics, list):
            self.metrics = [self.metrics]
        return {m.__name__: m(y_true, y_pred) for m in self.metrics}

    # TODO I need to redo this to work with my new approach
    @staticmethod
    def classification_threshold_report(y_true, y_pred, filename, extra_data=None):
        from metrics import ppv, npv, accuracy, balanced_accuracy
        import numpy as np

        nums_active = []

        thresholds = np.arange(0, 1, 0.01)

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

        ppv_mean = np.mean(np.array(all_ppvs), axis=0)
        ppv_std = np.std(np.array(all_ppvs), axis=0)

        npv_mean = np.mean(np.array(all_npvs), axis=0)
        npv_std = np.std(np.array(all_npvs), axis=0)

        accuracy_mean = np.mean(np.array(all_accuracies), axis=0)
        accuracy_std = np.std(np.array(all_accuracies), axis=0)

        balanced_accuracy_mean = np.mean(np.array(all_balanced_accuracies), axis=0)
        balanced_accuracy_std = np.std(np.array(all_balanced_accuracies), axis=0)

        num_active_mean = np.mean(np.array(all_nums_active), axis=0)
        num_active_std = np.std(np.array(all_nums_active), axis=0)

        import matplotlib.pyplot as plt

        plt.figure()
        fig, ax2 = plt.subplots()
        ax1 = ax2.twinx()
        ax2.plot(thresholds, ppv_mean, label="PPV")
        ax2.fill_between(thresholds, ppv_mean - ppv_std, ppv_mean + ppv_std, alpha=0.3)
        ax2.plot(thresholds, npv_mean, label="NPV")
        ax2.fill_between(thresholds, npv_mean - npv_std, npv_mean + npv_std, alpha=0.3)
        ax2.plot(thresholds, accuracy_mean, label="Accuracy")
        ax2.fill_between(thresholds, accuracy_mean - accuracy_std, accuracy_mean + accuracy_std, alpha=0.3)
        ax2.plot(thresholds, balanced_accuracy_mean, label="Balanced Accuracy")
        ax2.fill_between(thresholds, balanced_accuracy_mean - balanced_accuracy_std,
                         balanced_accuracy_mean + balanced_accuracy_std, alpha=0.3)
        ax1.plot(thresholds, num_active_mean, color="red", linestyle="--")
        print(num_active_mean)
        print(num_active_std)
        # ax2.fill_between(thresholds, num_active_mean - num_active_std, num_active_mean + num_active_std, alpha = 0.3, color = "red")
        ax2.fill_between(thresholds, num_active_mean - num_active_std, num_active_mean + num_active_std, alpha=1,
                         color="red")
        ax2.yaxis.label.set_text("Value of Metric")
        ax1.yaxis.label.set_text("--- Number of predicted actives")
        ax1.yaxis.label.set_color("red")
        ax2.set_xlabel("Classification Threshold")
        ax1.spines['right'].set_color('red')
        ax1.tick_params(axis='y', colors='red')
        plt.title("Validation Stats as a Function of Changing Threshold")
        ax2.set_xticks(np.arange(0, 1.1, 0.1))
        ax2.set_yticks(np.arange(0, 1.1, 0.1))
        ax2.set_xlim((0, 1))
        ax2.set_ylim((0, 1))
        ax1.set_xlim((0, 1))
        ax1.set_ylim((0, max(nums_active)))
        ax2.grid(visible=True, axis='both', alpha=0.5)
        ax2.legend(bbox_to_anchor=(-0.1, 1), loc='upper right', ncol=1)

        if extra_data:
            ax2.text(-0.5, 0.5, extra_data, fontsize=8)
        plt.savefig(filename, bbox_inches="tight", dpi=300)