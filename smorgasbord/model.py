import numpy as np
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
import torch
from torch import nn
import torch.nn.functional as F

# TODO instead of a wrapper class could I just set attributes of the passed model?
#  not to self I think using the sklearn api and building custom models from a given base class should suffice

# TODO need to add wrapper calls for all sklearn models that we want to support by default with auto loading plates
from qsar_utils import modi


def default(dictionary, key, default_val):
    if key not in dictionary.keys():
        return default_val
    else:
        return dictionary[key]


class XYDataset(Dataset):
    """
    Generic torch dataset for a dataframe to use with a loader
    """
    def __init__(self, X, y):
        self.y = y
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class XDataset(Dataset):
    """
    Generic torch dataset for a dataframe to use with a loader
    """
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


class QSARModel:
    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_probability(self, X):
        raise NotImplementedError

    def get_params(self):
        raise NotImplementedError

    def is_classifier(self):
        raise NotImplementedError

    def is_regressor(self):
        raise NotImplementedError

    def __str__(self):
        return ";".join([f"{key}:{val}" for key, val in self._args.items()])


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

    def is_classifier(self):
        return True

    def is_regressor(self):
        return False


class RandomForestRegressor(QSARModel):
    def __init__(self, **kwargs):
        from sklearn.ensemble import RandomForestRegressor as RFR
        self._args = kwargs
        self.model = RFR(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self):
        return self._args

    def is_classifier(self):
        return False

    def is_regressor(self):
        return True


class DecisionTreeClassifier(QSARModel):
    def __init__(self, **kwargs):
        from sklearn.tree import DecisionTreeClassifier as DTC
        self._args = kwargs
        self.model = DTC(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_probability(self, X):
        return self.model.predict_proba(X)

    def get_params(self):
        return self._args

    def is_classifier(self):
        return True

    def is_regressor(self):
        return False


class DecisionTreeRegressor(QSARModel):
    def __init__(self, **kwargs):
        from sklearn.tree import DecisionTreeRegressor as DTR
        self._args = kwargs
        self.model = DTR(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self):
        return self._args

    def is_classifier(self):
        return False

    def is_regressor(self):
        return True


class MLP(QSARModel, nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self._args = kwargs
        self.norm = nn.LayerNorm
        self.layer1 = nn.Linear
        self.layer2 = nn.Linear
        self.layer3 = nn.Linear

        self.optim = torch.optim.Adam(self.parameters(), lr=default(self._args, "lr", 0.003))
        self.loss = None

    def fit(self, X, y):
        self.train()
        self._args["input_dim"] = X.shape[1]
        self._args["output_dim"] = y.shape[1] if len(y.shape) > 1 else 1
        self.norm = self.norm(self._args["input_dim"])
        self.layer1 = self.layer1(self._args["input_dim"], self._args["hidden_dim"])
        self.layer2 = self.layer2(self._args["hidden_dim"], self._args["hidden_dim"])
        self.layer3 = self.layer3(self._args["hidden_dim"], self._args["output_dim"])

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loader = DataLoader(XYDataset(X, y), batch_size=default(self._args, "batch_size", X.shape[0]))

        training_loss = []
        for epoch in range(default(self._args, "epochs", 300)):
            for batch in loader:
                output = self(batch[0].float().to(device))
                l = self.loss(output, batch[1].float().to(device))
                self.optim.zero_grad()
                l.backward()
                self.optim.step()
                training_loss.append(l.item)

    def forward(self, x):
        x = self.norm(x)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        return x

    def predict(self, X):
        self.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loader = DataLoader(XDataset(X), batch_size=default(self._args, "batch_size", X.shape[0]))

        with torch.no_grad():
            running_output = np.array([])
            for batch in loader:
                running_output = np.concatenate(
                    (running_output, self(batch[0].float().to(device)).cpu().detach().numpy()))

        return running_output

    def get_params(self):
        return self._args

    def is_classifier(self):
        return True

    def is_regressor(self):
        return True


class MLPClassifier(MLP, nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss = nn.BCEWithLogitsLoss()

    def is_regressor(self):
        return False


class MLPRegressor(MLP, nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss = nn.MSELoss()

    def is_classifier(self):
        return False


class MODIdeep(QSARModel, nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self._args = kwargs

        self.norm = nn.LayerNorm
        self.layer1 = nn.Linear
        self.layer2 = nn.Linear

        self.loss = nn.BCEWithLogitsLoss()
        self.optim = torch.optim.Adam(self.parameters(), lr=default(self._args, "lr", 0.003))

    def forward(self, x):
        x = self.norm(x)
        x1 = self.layer1(x)
        x2 = F.relu(x1)
        x3 = self.layer2(x2)

        return x, x2, x3

    def latent_modi(self, x, y):
        x = self.norm(x)
        x = self.layer1(x)

        pre_activation_modi = modi(x, y)

        x = F.relu(x)

        post_activation_modi = modi(x, y)

        return pre_activation_modi, post_activation_modi

    def fit(self, X, y):
        self.train()
        self._args["input_dim"] = X.shape[1]
        self._args["output_dim"] = y.shape[1] if len(y.shape) > 1 else 1
        self.norm = self.norm(self._args["input_dim"])
        self.layer1 = self.layer1(self._args["input_dim"], self._args["hidden_dim"])
        self.layer2 = self.layer2(self._args["hidden_dim"], self._args["output_dim"])

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataset = XYDataset(X, y)
        loader = DataLoader(dataset, batch_size=default(self._args, "batch_size", X.shape[0]))

        self.to(device)

        for epoch in range(300):
            for i, batch in enumerate(loader):
                outputs = self.step(batch=batch[0].to(device))

                l = self.loss(outputs, batch[1].to(device))

                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()

    def predict(self, X):
        self.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataset = XDataset(X)
        loader = DataLoader(dataset, batch_size=default(self._args, "batch_size", X.shape[0]))

        self.to(device)

        preds = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                self.train()
                outputs = self.step(batch=batch["X"])
                preds += F.sigmoid(outputs).cpu().detach().tolist()

        return np.array(preds)