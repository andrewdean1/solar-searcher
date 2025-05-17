from sklearn.model_selection import train_test_split
from torch.nn import Conv2d, MaxPool2d, Parameter
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from torch.nn.functional import relu
import torch.optim as optim
from torch.nn import ReLU, Sigmoid
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import sys

# Main Linear Regression class
class LM:
    
    def __init__(self, all_feats = False): # "all_feats" arg. indicates if all or a subset of features are used
        
        # For running on GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu" 
        
        # Initializing/locating a model
        self.model = self.LinearModel(all_feats).to(self.device) # Pass feature usage info to implicit linear model object
    
    # Linear Model sub-class
    class LinearModel(nn.Module):
        
        def __init__(self, all_feats = False): # "all_feats" arg. passed from above
            
            # Initialize nn.Module object
            super().__init__()

            # Matrix alignment depending on if all features are used
            if all_feats:
                init_feats = 13
            else:
                init_feats = 12

            # Basic linear model pipeline
            self.pipeline = nn.Sequential(
                nn.Linear(init_feats, 10),
                ReLU(),
                nn.Linear(10, 6),
                ReLU(),
                nn.Linear(6,2),
                ReLU(),
                nn.Linear(2,1)
            )

        # Method to computes the scores/predictions
        def forward(self, x):
            return self.pipeline(x)
    
    # Data preprocessing/batching
    def preprocess_data(self, X_train, y_train, X_test, y_test):
        
        ## TODO: Incorporate data scaling or feature maps?
        
        # Training data batches
        data_loader_train = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size = 32,
        shuffle = True
        )

        # Testing data batches
        data_loader_test = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_test, y_test),
            batch_size = 32,
            shuffle = True
        )
        return data_loader_train, data_loader_test
    
    # Model training method
    def train(self, X, y, X_t, y_t, epochs = 10, verbose = True, **opt_kwargs):
        
        # Loss function is Mean-Squared-Error
        loss_fn = nn.MSELoss()

        # Optimizer is SGD with momentum
        optimizer = optim.SGD(self.model.parameters(), lr = 0.01, **opt_kwargs)

        # Bookkeeping arrays and min-loss var
        losses_train = []
        losses_test = []
        
        # Batching the data
        data_loader_train, data_loader_test = self.preprocess_data(X, y, X_t, y_t)
        
        # Train over n epochs
        for epoch in range(epochs):
            
            # Set model back to train mode
            self.model.train()

            # Iterate through each data batch
            for data in data_loader_train:
                
                # Locate data batch
                X_batch, y_batch = data
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                # Reset gradients
                optimizer.zero_grad()
                
                # Compute model predictions and loss
                y_pred_tr = self.model(X_batch)
                loss = loss_fn(y_pred_tr, y_batch.unsqueeze(1))
                
                # Compute gradients and optimize
                loss.backward()
                optimizer.step()

            # Record current model training/testing loss
            self.model.eval() # Set model to evaluation mode
            X, X_t, y, y_t = X.to(self.device), X_t.to(self.device), y.to(self.device), y_t.to(self.device)
            y_pred_tr = self.model(X)
            y_pred_te = self.model(X_t)
            loss_tr = loss_fn(y_pred_tr, y.unsqueeze(1))
            loss_te = loss_fn(y_pred_te, y_t.unsqueeze(1))

            if False:
                with torch.no_grad():
                    
                    # Compute test loss
                    test_losses = []
                    for X_batch, y_batch in data_loader_test:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        y_pred = self.model(X_batch)
                        test_loss = loss_fn(y_pred, y_batch.unsqueeze(1))
                        test_losses.append(test_loss.item())
                    loss_test = np.mean(test_losses)

            # Store and display current training loss
            losses_train.append(loss_tr.item())
            losses_test.append(loss_te.item())
            if verbose:
                print('Epoch: {} | Loss: {}'.format(epoch + 1, loss_te.item()))

        # Store training/testing losses    
        self.losses = losses_train
        self.test_loss = losses_test
    
    # Additional method to compute model predictions
    def predict(self, X, unscale=True):
        return self.model(X)

    # Method to compute model loss
    def loss(self):
        return self.losses, self.test_loss
    
    # Method to store model waits
    def saveModel(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        print('Saved model at {}'.format(filepath))
    
    # Method to retrieve stored model
    def loadModel(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
        self.model.eval()
    