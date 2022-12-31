import pickle
from scipy.stats import norm
import scipy.optimize as opt
import yfinance as yf
import pandas as pd
import pandas.core
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.svm import SVR
from scipy.stats import uniform as sp_rand
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import tensorflow.keras as keras
from keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, BatchNormalization, Activation, LeakyReLU
from keras.optimizers import Nadam
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras import backend as K
from get_data2 import GetData
from xgboost import XGBRFRegressor
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from test1 import NN

# from darts.models import NBEATSModel

"""Get data"""

data_feed = GetData()
df, X_train1, X_test1, Y_train1, Y_test1, scaler1 = data_feed.get_seq_data()
df2, X_train2, X_test2, Y_train2, Y_test2, scaler2 = data_feed.get_vec_data()

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

test_model = NN(input_size=X_test2.shape[1], output_size=1)
test_model = test_model.to(device=device)
optimizer = optim.Adam(test_model.parameters(), lr=0.001)

# load saved states onto newly initialized model and optimizer
test_model.load_state_dict(torch.load('NN.pytorch'))
optimizer.load_state_dict(torch.load('NN_optimizer.pytorch'))
test_model.eval()



X_test2 = X_test2.float().to(device=device)
X_test2 = X_test2.reshape(X_test2.shape[0], -1)
x = test_model(X_test2)
y = Y_test2.float().to(device=device)
y = y.reshape(y.shape[0], -1)
# print(pred)
# print(Y_test2.float.reshape(Y_test2.shape[0], -1))
plt.plot(y.tolist(), color='r', label='real')
plt.plot(x.tolist(), color='b', label='prediction')
plt.legend()
plt.show()