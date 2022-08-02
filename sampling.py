class Sampler:
    def __init__(self, cache=False, random_state=None):
        self._random_state = random_state
        self._cache = cache

    def undersample(self, X, y):
        from imblearn.under_sampling import RandomUnderSampler

        ros = RandomUnderSampler(random_state=self._random_state)
        X_res, y_res = ros.fit_resample(X, y)

        return X_res, y_res

    def oversample(self, X, y):
        from imblearn.over_sampling import RandomOverSampler

        ros = RandomOverSampler(random_state=self._random_state)
        X_res, y_res = ros.fit_resample(X, y)

        return X_res, y_res

    def remove_most_similar(self, X, y):
        raise NotImplementedError

    def remove_least_similar(self, X, y):
        raise NotImplementedError

    def smote(self, X, y):
        from imblearn.over_sampling import SMOTE

        sm = SMOTE(random_state=self._random_state)
        X_res, y_res = sm.fit_resample(X, y)

        return X_res, y_res

    def add_custom_sampling_func(self, name, func):
        self.__setattr__(str(name), func)

    def func_exists(self, name):
        func_call = "calc_" + str(name)
        return func_call in dir(self) and callable(self.__getattribute__(func_call))

    def get_sampling_func(self, name):
        if self.func_exists(name):
            return self.__getattribute__("calc_" + name)
        else:
            raise ValueError(f"sampling function {name} does not exist")
