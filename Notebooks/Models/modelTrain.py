#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch 
from Notebooks.Models.models import LM
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    
    data = pd.read_csv("../../Data/csvs/full_data.csv")
    data  = data.drop('irradiance', axis = 1)
    y = data['pvo']
    X = data.drop('pvo', axis = 1)
    for col in X.columns:
        X[col] = (X[col] - X[col].mean())/X[col].std()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    X_train = torch.tensor(X_train.values, dtype = torch.float32)
    y_train = torch.tensor(y_train.values, dtype = torch.float32)
    X_test = torch.tensor(X_test.values, dtype = torch.float32)
    y_test = torch.tensor(y_test.values, dtype = torch.float32)

    model = LM()
    model.train(X_train, y_train, X_test, y_test, epochs = 1000)

    tr_loss, te_loss = model.loss()
    tr = np.array(tr_loss).reshape(-1,1)
    te = np.array(te_loss).reshape(-1,1)
    
    np.savetxt('loss.txt', np.concatenate((tr, te), axis = 1))
    model.saveModel('LM.pt.tar')
    
if __name__ == "__main__":
    main()