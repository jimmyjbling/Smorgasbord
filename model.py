class QSARModel:
    def __init__(self, parent, model, name, child, child_name, desc_name, desc_settings, label):
        self.model = model
        self.name = name
        self.desc_name = desc_name
        self.desc_settings = desc_settings
        self.parent = parent
        self.child_name = child_name
        self.child = child
        self.label = label
        self.screening_stats = {}


    def fit():
        raise NotImplementedError

    def predict():
        raise NotImplementedError

    def predict_probability():
        raise NotImplementedError

class RF(QSARModel):

    def __init__(self, n_estimators = 100, **kwargs):

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import RandomForestRegressor

        self.model = RandomForestClassifier(n_estimators = n_estimators, **kwargs)

    def fit(self, x, y):

        self.model.fit(x, y)

    def predict_probability(self, x):

        active_probability = (self.model.predict_proba(x)[:, 1])
        return active_probability

