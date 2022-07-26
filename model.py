from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
import torch
from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import RMSprop, SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from torch.nn.parallel import DataParallel

import numpy as np



# TODO instead of a wrapper class could I just set attributes of the passed model?

class QSARModel:
    '''
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
    '''

    def fit():
        raise NotImplementedError

    def predict():
        raise NotImplementedError

    def predict_probability():
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError

    def to_string(self):
        #should return a friendly (lowercase with underscores) description of model name + parameters
        #e.g. random_forest_100 for rf with 100 trees

        raise NotImplementedError

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

'''
class VanillaNN(QSARModel):

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
'''
 
class VanillaNN(nn.Module):


    def __init__(self, input_shape):

        super(VanillaNN, self).__init__()
         
        self.fc1 = nn.Linear(input_shape, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 10)
        self.fc6 = nn.Linear(10, 10)
        self.fc7 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

        self.d1 = nn.Dropout(0.55)
        #self.d2 = nn.Dropout(0.1)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        #x = self.d1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.d1(x)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        #x = self.d2(x)
        x = F.relu(self.fc6(x))
        #x = self.d1(x)
        x = self.fc7(x)
        x = self.sigmoid(x)
        return x

class NN(QSARModel):

    def __init__(self):
        self.name = "NN"

        pass

    def reset(self):
        self.model = None

    def fit(self, x, y):

        self.model = VanillaNN(input_shape = x.shape[1])
 
        train_x = torch.tensor(x, dtype=torch.float).cuda()
        train_y = torch.tensor(y, dtype=torch.float).cuda()

        train_y = train_y.unsqueeze(dim = 1)

        self.model = self.model.cuda()
        criterion = nn.BCELoss()

        learning_rate = 0.001

        optimizer = optim.Adam(self.model.parameters(), lr = learning_rate)

        num_epochs = 300

        #train_loop
        self.model.train()
        for _ in range(num_epochs):
            #print(f"\rEpoch {_}", end = "")

            pred = self.model.forward(train_x)
            loss = criterion(pred, train_y)
            #print("Loss: ", loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



    def predict_probability(self, x):

        test_x = torch.tensor(x, dtype=torch.float).cuda()

        self.model.eval()
        with torch.no_grad():
            prediction = self.model.forward(test_x).cpu()

        prediction = np.array(prediction.flatten())
        return prediction

    def get_string(self):

        return "NN"


    '''
    def save(self, filename):
        if not self.model:
            raise Exception("Can't save model before fitting (model parameters depend on descriptor length)")

        torch.save(filename)

    @classmethod
    def from_file(cls, filename):


        m = torch.load(filename
    '''






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
