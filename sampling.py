class Sampler:
    def __init__(self):
        raise NotImplementedError

    def undersample(self):
        raise NotImplementedError

    def oversample(self):
        raise NotImplementedError

    def remove_most_similar(self):
        raise NotImplementedError

    def remove_least_similar(self):
        raise NotImplementedError

    def smote(self):
        raise NotImplementedError
