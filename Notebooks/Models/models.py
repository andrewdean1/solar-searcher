import torch
import pandas as pd
from matplotlib import pyplot as plt
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, Parameter
from torch.nn.functional import relu
import torch.optim as optim
from torch.nn import ReLU
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
plt.style.use('seaborn-v0_8-whitegrid')


class LM:
    def __init__(
            self,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.LinearModel().to(self.device)
    class LinearModel(nn.Module):
        def __init__(self):
            super().__init__()

            self.pipeline = nn.Sequential(
                # nn.Flatten(),
                nn.Linear(12, 10),
                ReLU(),
                nn.Linear(10, 6),
                ReLU(),
                nn.Linear(6,2),
                ReLU(),
                nn.Linear(2,1)
            )

        # this is the customary name for the method that computes the scores
        # the loss is usually computed outside the model class during the training loop
        def forward(self, x):
            return self.pipeline(x)
    
    def preprocess_data(self, X_train, y_train, X_test, y_test):
        ## TODO: Input data scaling?
        data_loader_train = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size = 32,
        shuffle = True
        )

        data_loader_test = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_test, y_test),
            batch_size = 32,
            shuffle = True
        )
        return data_loader_train, data_loader_test
    

    def train(self, X, y, X_t, y_t, epochs = 10, **opt_kwargs):

        # loss function is cross-entropy (multiclass logistic)
        loss_fn = nn.MSELoss()

        # optimizer is SGD with momentum
        optimizer = optim.SGD(self.model.parameters(), lr = 0.001, **opt_kwargs)

        losses_train = []
        losses_test = []
        

        data_loader_train, data_loader_test = self.preprocess_data(X,y, X_t, y_t)
        # test_loader_iter = iter(data_loader_test)
        for epoch in range(epochs):
            for data, te_data in zip(data_loader_train, data_loader_test):
                X, y = data
                X, y = X.to(self.device), y.to(self.device)
                X_t, y_t = te_data
                X_t, y_t = X_t.to(self.device), y_t.to(self.device)
                # clear any accumulated gradients
                optimizer.zero_grad()

                # compute the loss
                y_pred = self.model(X)
                loss = loss_fn(y_pred, y)
                y_test_pred = self.model(X_t)
                loss_test = loss_fn(y_test_pred, y_t)
                # compute gradients and carry out an optimization step
                loss.backward()
                optimizer.step()
            print('Epoch: {} Loss: {}'.format(epoch + 1, loss))
            losses_train.append(loss.item())
            losses_test.append(loss_test.item())
        self.losses = losses_train
        self.test_loss = losses_test
    
    def predict(self, X):
        return self.model(X)

    def loss(self):
        return self.losses, self.test_loss
    def saveModel(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        print('Saved model at {}'.format(filepath))
    def loadModel(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
        self.model.eval()
    
class CNN(LM):
    def __init__(self,):
        super().__init__()
        self.model = self.CNN().to(self.device)
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()

            self.pipeline = nn.Sequential(
                # nn.Flatten(),
                nn.Linear(16, 10),
                ReLU(),
                nn.Linear(10, 6),
                ReLU(),
                nn.Linear(6,2),
                ReLU(),
                nn.Linear(2,1)
            )
