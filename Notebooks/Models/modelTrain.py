#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from models import LM
import pandas as pd
import numpy as np
import torch
import sys
import os

# Not needed since this file is in the same directory as models.py?
sys.path.append(os.path.abspath('../Models'))

def main():
    
    # Read in and prepare data
    data = pd.read_csv("../../Data/csvs/full_data.csv")
    data  = data.drop('irradiance', axis = 1)
    y = data['pvo']
    X = data.drop('pvo', axis = 1)
   
    # Standardize the data
    for col in X.columns:
        X[col] = (X[col] - X[col].mean()) / X[col].std()

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    X_train = torch.tensor(X_train.values, dtype = torch.float32)
    y_train = torch.tensor(y_train.values, dtype = torch.float32)
    X_test = torch.tensor(X_test.values, dtype = torch.float32)
    y_test = torch.tensor(y_test.values, dtype = torch.float32)

    # Initialize and train linear regression model
    model = LM()
    model.train(X_train, y_train, X_test, y_test, epochs = 20) # Increase number of epochs

    # Compute model train/test loss
    tr_loss, te_loss = model.loss()
    tr = np.array(tr_loss).reshape(-1,1)
    te = np.array(te_loss).reshape(-1,1)
    
    # Store model weights and train/test loss
    np.savetxt('../../Data/model/loss.txt', np.concatenate((tr, te), axis = 1))
    model.saveModel('../../Data/model/LM.pt.tar')
    
if __name__ == "__main__":
    main()