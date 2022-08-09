from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
import torch
from torch import nn

# TODO instead of a wrapper class could I just set attributes of the passed model?
#  not to self I think using the sklearn api and building custom models from a given base class should suffice

# TODO need to add wrapper calls for all sklearn models that we want to support by default with auto loading plates


class QSARModel:
    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_probability(self, X):
        raise NotImplementedError

    def get_params(self):
        raise NotImplementedError


class RandomForestClassifier(QSARModel):
    def __init__(self, **kwargs):
        from sklearn.ensemble import RandomForestClassifier as RFC
        self._args = kwargs
        self.model = RFC(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_probability(self, X):
        return self.model.predict_proba(X)

    def get_params(self):
        return self._args


class RF(QSARModel):

    def __init__(self, n_estimators = 100, **kwargs):

        self.stored_args = {}
        self.stored_args["n_estimators"] = n_estimators
        for key, value in kwargs.items():
            self.stored_args[key] = value

        self.name = "Random Forest"


        from sklearn.ensemble import RandomForestClassifier

        self.model = RandomForestClassifier(n_estimators = n_estimators, **kwargs)

    def fit(self, x, y):

        self.model.fit(x, y)

    def predict_probability(self, x):

        try:
            active_probability = (self.model.predict_proba(x)[:, 1])
        except:
            active_probability = (1 - self.model.predict_proba(x)[:, 0])

        return active_probability

    def to_dict(self):
        d = {}
        d["Name"] = self.name
        d["Arguments"] = self.stored_args
        return d

    def get_string(self):

        return "random_forest_" + str(self.stored_args["n_estimators"])
class XYDataset(Dataset):
    def __init__(self, X, y):
        self.y = y
        self.X = X

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        return X[idx], label[idx]


class MODIdeep:
    def __init__(self, in_feats, hidden_feats, out_feats=1):
        super().__init__()
        self.norm = nn.LayerNorm(in_feats)
        self.layer1 = nn.Linear(in_feats, hidden_feats)
        self.layer2 = nn.Linear(hidden_feats, out_feats)
        self.loss = nn.BCEWithLogitsLoss
        self.optimizer = nn.Adam()

    def forward(self, x):
        x = self.norm(x)
        x = self.layer1(x)
        x = nn.ReLU(x)
        x = self.layer2(x)

        return x

    def latent_modi(self, x, y):
        x = self.norm(x)
        x = self.layer1(x)

        pre_activation_modi = modi(x, y)

        x = nn.ReLU(x)

        post_activation_modi = modi(x, y)

        return pre_activation_modi, post_activation_modi

    def fit(self, X, y):
        dataset = XYDataset(X, y)
        loader = DataLoader(dataset, batch_size=200)
        self.train()
        for epoch in range(300):
            for i, batch in enumerate(loader):
                outputs = self.step(batch=batch[0])

                l = loss(outputs, batch[1])

                optimizer.zero_grad()
                l.backward()
                optimizer.step()

            if epoch % 25 == 0:
                print(f"{epoch} loss: {l.item()}")

    def predict(self, X, y):
        dataset = XYDataset(X, y)
        loader = DataLoader(dataset, batch_size=200)
        self.eval()
        preds = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                self.train()
                outputs = self.step(batch=batch["X"])
                preds += nn.functional.sigmoid(outputs).tolist()
                l = loss(outputs, batch["y"], batch["num_lig_atoms"])
        return stats
