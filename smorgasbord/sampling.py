import numpy as np


class Sampler:
    def __init__(self, random_state=None):
        self._random_state = random_state

    def calc_undersample(self, X, y, return_mask=False):
        from imblearn.under_sampling import RandomUnderSampler

        rus = RandomUnderSampler(random_state=self._random_state)
        X_res, y_res = rus.fit_resample(X, y)

        if return_mask:
            return self._flip_index(y.shape[0], rus.sample_indices_)
        else:
            return X_res, y_res

    def calc_oversample(self, X, y, return_mask=False):
        from imblearn.over_sampling import RandomOverSampler

        ros = RandomOverSampler(random_state=self._random_state)
        X_res, y_res = ros.fit_resample(X, y)

        if return_mask:
            return self._flip_index(y.shape[0], ros.sample_indices_)
        else:
            return X_res, y_res

    def calc_remove_most_similar(self, X, y, return_mask=False):
        raise NotImplementedError

    def calc_remove_least_similar(self, X, y, return_mask=False):
        raise NotImplementedError

    @staticmethod
    def _get_func_name(mask_name):
        if not mask_name.startswith("calc_"):
            mask_name = f"calc_{mask_name}"
        return mask_name

    def _flip_index(self, size, index):
        mask = np.full(size, True, bool)
        mask[index] = False
        index = np.arange(size)
        return index[mask]

    def add_sampling_func(self, mask_name, func):
        self.__setattr__(self._get_func_name(mask_name), func)

    def func_exists(self, mask_name):
        return self._get_func_name(mask_name) in dir(self) and callable(self.__getattribute__(self._get_func_name(mask_name)))

    def get_func(self, mask_name):
        if self.func_exists(mask_name):
            return self.__getattribute__(self._get_func_name(mask_name))
        else:
            raise ValueError(f"sampling function {mask_name} does not exist")

    def get_available_funcs(self):
        return [x.startswith("calc_") and callable(self.__getattribute__(x)) for x in dir(self)]


class DatasetSampler(Sampler):
    def __init__(self, cache=True):
        self._cache = cache
        super().__init__()

    def mask_exist(self, mask_name):
        return mask_name in dir(self)

    def _calc_sample(self, mask_name, X, y):
        mask = self.get_func(mask_name)(X, y, return_mask=True)
        if self._cache:
            self.__setattr__(mask_name, mask)
        return mask

    def get_mask(self, mask_name, X=None, y=None):
        if self.mask_exist(mask_name):
            return self.__getattribute__(mask_name)
        else:
            if self.func_exists(mask_name) and X is not None and y is not None:
                return self._calc_sample(mask_name, X, y)
            else:
                raise ValueError(f"mask {mask_name} does not exist")

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

