class Procedure:
    def __init__(self, random_state=None):
        self._random_state = random_state

    def screen(self, model, screening_X, predict_prob=False):
        if predict_prob:
            return model.predict_proba(screening_X)
        else:
            return model.predict(screening_X)

    def train(self, model, training_dataset):
        pass

    def cross_validate(self, model, training_dataset, cv=None):
        pass
