from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
import torch
from torch import nn


# TODO instead of a wrapper class could I just set attributes of the passed model?

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
