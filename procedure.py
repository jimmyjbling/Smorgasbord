from metrics import get_classification_metrics


class Procedure:
    def __init__(self, random_state=None):
        self._random_state = random_state

    @staticmethod
    def screen(model, screening_X, predict_prob=False):
        if predict_prob:
            return model.predict_proba(screening_X)
        else:
            return model.predict(screening_X)

    @staticmethod
    def train(model, train_X, train_y):
        model.fit(train_X, train_y)

    def cross_validate(self, model, train_X, train_y, cv=None, **kwargs):
        from copy import deepcopy
        if cv is None:
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold

        kwargs["random_state"] = self._random_state

        s = cv(**kwargs)

        cv_models = {}

        for train_index, test_index in s.split(train_X, train_y):
            X_train, X_test = train_X[train_index], train_X[test_index]
            y_train, y_test = train_y[train_index], train_y[test_index]

            model_copy = deepcopy(model)

            self.train(model_copy, X_train, y_train)
            y_pred = self.screen(model, X_test, predict_prob=False)

            cv_models[model_copy] = (y_test, y_pred)

        return cv_models

    def eval(self, y_pred, y_true, metrics=None):
        if metrics is None:
            if y_true.dtype == int:
                return get_classification_metrics(y_true, y_pred)
            else:
                raise NotImplementedError("no continuous metrics yet")
        else:
            if not isinstance(metrics, list):
                metrics = [metrics]
            return {m.__name__: m(y_true, y_pred) for m in metrics}
