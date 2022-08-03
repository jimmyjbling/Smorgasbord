class Sampler:
    def __init__(self, cache=False, random_state=None):
        self._random_state = random_state
        self._cache = cache

    def undersample(self, X, y, return_mask=False):
        from imblearn.under_sampling import RandomUnderSampler

        rus = RandomUnderSampler(random_state=self._random_state)
        X_res, y_res = rus.fit_resample(X, y)

        if return_mask:
            return rus.sample_indices_
        else:
            return X_res, y_res

    def oversample(self, X, y, return_mask=False):
        from imblearn.over_sampling import RandomOverSampler

        ros = RandomOverSampler(random_state=self._random_state)
        X_res, y_res = ros.fit_resample(X, y)

        if return_mask:
            return ros.sample_indices_
        else:
            return X_res, y_res

    def remove_most_similar(self, X, y, return_mask=False):
        raise NotImplementedError

    def remove_least_similar(self, X, y, return_mask=False):
        raise NotImplementedError

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

    # def get_masks(self):
    #     return self._masks
    #
    # def get_mask_names(self):
    #     return self._masks.keys()
    #
    # def get_mask(self, mask_name):
    #     return self._masks[mask_name]
    #
    # def get_masked_dataset(self, mask_name):
    #     if mask_name in self._masks.keys():
    #         mask_name = self._masks[mask_name]
    #     return self.dataset.loc[mask_name]
    #
    # def iter_masks(self):
    #     for key, val in self._masks.items():
    #         yield key, val
    #
    # def add_mask(self, name, indices):
    #     self._masks[name] = indices
    #
    # def remove_mask(self, name):
    #     if name in self._masks.keys():
    #         del self._masks[name]