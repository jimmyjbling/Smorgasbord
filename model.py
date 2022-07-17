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
