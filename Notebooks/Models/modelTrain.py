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
    data_no_irr  = data.drop("irradiance", axis = 1)
    y_no_irr = data_no_irr['pvo']
    X_no_irr = data_no_irr.drop('pvo', axis = 1)
    y_irr = data["pvo"]
    X_irr = data.drop('pvo', axis = 1)
   
    # Standardize the data (w/out irr)
    for col in X_no_irr.columns:
        X_no_irr[col] = (X_no_irr[col] - X_no_irr[col].mean()) / X_no_irr[col].std()

    # Standardize the data (w/ irr)
    for col in X_irr.columns:
        X_irr[col] = (X_irr[col] - X_irr[col].mean()) / X_irr[col].std()

    # Split train/test (w/out irr)
    X_train_no_irr, X_test_no_irr, y_train_no_irr, y_test_no_irr = train_test_split(X_no_irr, y_no_irr, test_size = 0.3)
    X_train_no_irr = torch.tensor(X_train_no_irr.values, dtype = torch.float32)
    y_train_no_irr = torch.tensor(y_train_no_irr.values, dtype = torch.float32)
    X_test_no_irr = torch.tensor(X_test_no_irr.values, dtype = torch.float32)
    y_test_no_irr = torch.tensor(y_test_no_irr.values, dtype = torch.float32)

    # Split train/test (w/ irr)
    X_train_irr, X_test_irr, y_train_irr, y_test_irr = train_test_split(X_irr, y_irr, test_size = 0.3)
    X_train_irr = torch.tensor(X_train_irr.values, dtype = torch.float32)
    y_train_irr = torch.tensor(y_train_irr.values, dtype = torch.float32)
    X_test_irr = torch.tensor(X_test_irr.values, dtype = torch.float32)
    y_test_irr = torch.tensor(y_test_irr.values, dtype = torch.float32)

    # Initialize and train linear regression model (w/out irr)
    print("Model 1:")
    model1 = LM()
    model1.train(X_train_no_irr, y_train_no_irr, X_test_no_irr, y_test_no_irr, epochs = 5) # Increase number of epochs

    print("\n----------\n")

    # Initialize and train linear regression model (w/ irr)
    print("Model 2:")
    model2 = LM(all_feats = True)
    model2.train(X_train_irr, y_train_irr, X_test_irr, y_test_irr, epochs = 5) # Increase number of epochs

    # Compute model train/test loss (w/out irr)
    tr_loss_no_irr, te_loss_no_irr = model1.loss()
    tr_no_irr = np.array(tr_loss_no_irr).reshape(-1,1)
    te_no_irr = np.array(te_loss_no_irr).reshape(-1,1)

    # Compute model train/test loss (w/ irr)
    tr_loss_irr, te_loss_irr = model2.loss()
    tr_irr = np.array(tr_loss_irr).reshape(-1,1)
    te_irr = np.array(te_loss_irr).reshape(-1,1)
    
    # Store model weights and train/test loss (w/out irr)
    np.savetxt('../../Data/model/loss_no_irr.txt', np.concatenate((tr_no_irr, te_no_irr), axis = 1))
    model1.saveModel('../../Data/model/LM_no_irr.pt.tar')

    # Store model weights and train/test loss (w/ irr)
    np.savetxt('../../Data/model/loss_irr.txt', np.concatenate((tr_irr, te_irr), axis = 1))
    model2.saveModel('../../Data/model/LM_no_irr.pt.tar')
    
if __name__ == "__main__":
    main()