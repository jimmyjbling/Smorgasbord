class Sampler:
    def __init__(self, cache=False, random_state=None):
        self._random_state = random_state
        self._cache = cache

        self._funcs = ["undersample", "oversample"]

    def undersample(self, X, y, return_mask=False):
        if "undersample_mask" in self.__dict__:
            return self.__getattribute__("undersample_mask")

        from imblearn.under_sampling import RandomUnderSampler

        rus = RandomUnderSampler(random_state=self._random_state)
        X_res, y_res = rus.fit_resample(X, y)

        if self._cache:
            self.__setattr__("undersample_mask", rus.sample_indices_)

        if return_mask:
            return rus.sample_indices_
        else:
            return X_res, y_res

    def oversample(self, X, y, return_mask=False):
        if "oversample_mask" in self.__dict__:
            return self.__getattribute__("oversample_mask")

        from imblearn.over_sampling import RandomOverSampler

        ros = RandomOverSampler(random_state=self._random_state)
        X_res, y_res = ros.fit_resample(X, y)

        if self._cache:
            self.__setattr__("oversample_mask", ros.sample_indices_)

        if return_mask:
            return ros.sample_indices_
        else:
            return X_res, y_res

    def remove_most_similar(self, X, y, return_mask=False):
        raise NotImplementedError

    def remove_least_similar(self, X, y, return_mask=False):
        raise NotImplementedError

    def add_custom_func(self, name, func):
        self.__setattr__(str(name), func)
        self._funcs.append(name)

    def func_exists(self, name):
        return name in dir(self) and callable(self.__getattribute__(name))

    def get_func(self, name):
        if self.func_exists(name):
            return self.__getattribute__(name)
        else:
            raise ValueError(f"sampling function {name} does not exist")

    def get_available_funcs(self):
        return self._funcs

    def mask_exist(self, name):
        return f"{name}_mask" in self.__dict__

    def get_mask(self, name):
        if self.mask_exist(name):
            return self.__getattribute__(f"{name}_mask")
        else:
            raise ValueError(f"mask {name} does not exist")

    def get_available_masks(self):
        return [x for x in self.__dict__ if "_mask" in x]

    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, value):
        self._cache = bool(value)

    @cache.deleter
    def cache(self):
        for attr in self.__dict__.keys():
            if attr != "_cache":
                self.__setattr__(attr, None)

