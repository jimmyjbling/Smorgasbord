from metrics import get_default_classification_metrics, get_default_regression_metrics


class Procedure:
    def __init__(self, random_state=None):
        self._random_state = random_state

    def screen(self, model, screening_dataset, descriptor_func, dataset=None, sampling_func=None):

        screening_X = screening_dataset.get_descriptor(descriptor_func)

        if "predict_proba" in dir(model) and callable(model.__getattribute__("predict_proba")):
            res = model.predict_proba(screening_X)
        else:
            res = model.predict(screening_X)

        return res

    def train(self, model, dataset, descriptor_func, sampling_func):

        y = dataset.get_label(mask_name=sampling_func)
        X = dataset.get_descriptor(descriptor_func, mask_name=sampling_func)

        model.fit(X, y)

        return {model: None}

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
            res = model.predict_proba(X_test)
        else:
            res = model.predict(X_test)

        return {model: (y_test, res)}

    def cross_validate(self, model, dataset, descriptor_func, sampling_func, cv=None, **kwargs):
        from copy import deepcopy
        if cv is None:
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold

        kwargs["random_state"] = self._random_state

        s = cv(**kwargs)

        cv_models = {}

        y = dataset.get_label(mask_name=sampling_func)
        X = dataset.get_descriptor(descriptor_func, mask_name=sampling_func)

        for train_index, test_index in s.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model_copy = deepcopy(model)

            model_copy.fit(X, y)

            if "predict_proba" in dir(model) and callable(model.__getattribute__("predict_proba")):
                res = model.predict_proba(X_test)
            else:
                res = model.predict(X_test)

            cv_models[model_copy] = (y_test, res)

        return cv_models

    def eval(self, y_true, y_pred, metrics=None):
        if metrics is None:
            if y_true.dtype == int:
                metrics = get_default_classification_metrics()
            else:
                metrics = get_default_regression_metrics()

        if not isinstance(metrics, list):
            metrics = [metrics]
        return {m.__name__: m(y_true, y_pred) for m in metrics}
