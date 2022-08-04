from metrics import get_classification_metrics


class Procedure:
    def __init__(self, random_state=None):
        self._random_state = random_state

    def screen(self, model, screening_dataset, descriptor_func, dataset=None, sampling_func=None, predict_prob=False):

        screening_X = screening_dataset.get_descriptor(descriptor_func)

        if predict_prob:
            return model.predict_proba(screening_X)
        else:
            return model.predict(screening_X)

    def train(self, model, dataset, descriptor_func, sampling_func):

        y = dataset.get_label(mask_name=sampling_func)
        X = dataset.get_descriptor(descriptor_func, mask_name=sampling_func)

        model.fit(X, y)

        return {model: None}

    def train_with_test(self, model, dataset, descriptor_func, sampling_func, predict_prob=False, cv=None, **kwargs):

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
        if predict_prob:
            y_pred = model.predict_proba(X_test)
        else:
            y_pred = model.predict(X_test)

        return {model: (y_test, y_pred)}

    def cross_validate(self, model, dataset, descriptor_func, sampling_func, predict_prob=False, cv=None, **kwargs):
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
            if predict_prob:
                y_pred = model_copy.predict_proba(X_test)
            else:
                y_pred = model_copy.predict(X_test)

            cv_models[model_copy] = (y_test, y_pred)

        return cv_models

    def eval(self, y_true, y_pred, metrics=None):
        if metrics is None:
            if y_true.dtype == int:
                return get_classification_metrics(y_true, y_pred)
            else:
                raise NotImplementedError("no continuous metrics yet")
        else:
            if not isinstance(metrics, list):
                metrics = [metrics]
            return {m.__name__: m(y_true, y_pred) for m in metrics}
