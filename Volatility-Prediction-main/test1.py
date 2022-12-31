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
from get_data import GetData
from xgboost import XGBRFRegressor
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# from darts.models import NBEATSModel

"""Get data"""

data_feed = GetData()
df, X_train1, X_test1, Y_train1, Y_test1, scaler1, WINDOW_SIZE = data_feed.get_seq_data()

df2, X_train2, X_test2, Y_train2, Y_test2, scaler2 = data_feed.get_vec_data()





# """SVR"""
# SVR_model = pickle.load(open('SVR_pickle', 'rb'))
# # # Create different SVR models
# # # svr_poly = SVR(kernel='poly', degree=2)
# # svr_lin = SVR(kernel='linear')
# # # svr_rbf = SVR(kernel='rbf')

# # # Search for best hyperparameters
# # para_grid = {'gamma': sp_rand(),
# #           'C': sp_rand(),
# #           'epsilon': sp_rand()}
# # SVR_model = RandomizedSearchCV(svr_lin, para_grid)

# # # Fit and train the model on the data
# # print("fitting model")
# # SVR_model.fit(X_train2, Y_train2) # the y is basically the realized volatility but shifted right by 1, so the x contains the current volatility and squared returns to predict the next timestep's volatility

# # Prediction on the testing set
# SVR_predictions = SVR_model.predict(X_test2)
# SVR_predictions = pd.DataFrame(SVR_predictions)
# SVR_predictions = SVR_predictions[:-1]

# df2_test_copy = df2.copy()[-len(SVR_predictions):]
# SVR_predictions.index = df2_test_copy.index
# df2_test_copy['realized_vol'] = SVR_predictions[0]

# SVR_predictions = scaler2.inverse_transform(df2_test_copy)
# SVR_predictions = pd.DataFrame(SVR_predictions)

# SVR_predictions.index = df2_test_copy.index

# SVR_predictions = list(SVR_predictions[8])


# # Calculate RMSE
# rmse_svr = mse(Y_test2, SVR_predictions, squared=False) / 100
# print('The RMSE value of SVR with Linear Kernel is {:.6f}'.format(rmse_svr))

# # Plot out the prediction on top of the real value
# plt.figure(figsize=(10, 6))
# plt.plot(df2.index, [a/100 for a in df2['realized_vol']], label='Realized Volatility')
# plt.plot(df2_test_copy.index, [b/100 for b in SVR_predictions], label='Volatility Prediction-SVR-GARCH')
# plt.title('Volatility Prediction with SVR-GARCH (Linear)', fontsize=12)
# plt.legend()
# plt.show()

# # # Save the model
# # with open('SVR_pickle', 'wb') as f:
# #     pickle.dump(SVR_model, f)



# """Neural network"""

# class NN(nn.Module):
#     def __init__(self, input_size=X_train2.shape[1], output_size=1):
#         super(NN, self).__init__()
#         self.d1 = nn.Linear(input_size, 512)
#         self.leaky_relu = nn.LeakyReLU(0.05)
#         self.drop = nn.Dropout(0.2)
#         self.d2 = nn.Linear(512, 256)
#         self.d3 = nn.Linear(256, 256)
#         self.d4 = nn.Linear(256, 64)
#         self.d5 = nn.Linear(64, output_size)
    
#     def forward(self, x):
#         x = self.leaky_relu(self.d1(x))
#         x = self.drop(x)
#         x = self.leaky_relu(self.d2(x))
#         x = self.drop(x)
#         x = self.leaky_relu(self.d3(x))
#         x = self.drop(x)
#         x = self.leaky_relu(self.d4(x))
#         x = self.d5(x)

#         return x

# # Initialize device
# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# # Hyperparameters
# input_size = X_train2.shape[1]
# output_size = 1
# batch_size = 64
# learning_rate = 0.001
# num_epoch = 10


# # Load data
# train_loader = DataLoader(dataset=X_train2, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(dataset=X_test2, batch_size=batch_size, shuffle=False)
# X_train2 = torch.tensor(X_train2)
# Y_train2 = torch.tensor(Y_train2)
# X_test2 = torch.tensor(X_test2)
# Y_test2 = torch.tensor(Y_test2)

# # Initialize network
# model = NN().to(device)

# # Compile
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# # Train
# for epoch in range(num_epoch):
#     data = X_train2.float().to(device=device)
#     targets = Y_train2.float().to(device=device)
#     # print(data.shape)
#     # print(targets.shape)
#     data = data.reshape(data.shape[0], -1)
#     pred = model(data)
#     loss = criterion(pred, targets)

#     optimizer.zero_grad()
#     loss.backward()

#     optimizer.step()

# # Save model and optimizer state

# # torch.save(model.state_dict(), "NN.pytorch")
# # torch.save(optimizer.state_dict(), "NN_optimizer.pytorch")

# # Initialize new model and optimizer
# test_model = NN(input_size=X_test2.shape[1], output_size=1)
# test_model = test_model.to(device=device)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # load saved states onto newly initialized model and optimizer
# model.load_state_dict(torch.load('NN.pytorch'))
# optimizer.load_state_dict(torch.load('NN_optimizer.pytorch'))
# model.eval()


# X_test2 = X_test2.float().to(device=device)
# X_test2 = X_test2.reshape(X_test2.shape[0], -1)
# x = model(X_test2)
# y = Y_test2.float().to(device=device)
# y = y.reshape(y.shape[0], -1)
# # print(pred)
# # print(Y_test2.float.reshape(Y_test2.shape[0], -1))
# plt.plot(y.tolist(), color='r', label='real')
# plt.plot(x.tolist(), color='b', label='prediction')
# plt.legend()
# plt.show()


"""LSTM"""

# Hyperparameters
input_size = X_train1.shape[2] # number of features
sequence_length = WINDOW_SIZE # sequence length
output_size = 1
hidden_size = 256
batch_size = 64
num_layers = 4
learning_rate = 0.1
num_epoch = 10

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.leaky_relu = nn.ReLU()

    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=device) # x.size(0) is the batch size
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=device)
        out, _ = self.lstm1(x, (h0, c0))
        # out = out.reshape(out.shape[0], -1)
        out = self.fc1(out[:, -1, :]) 
        return out

# Initialize device



# Load data
# train_loader = DataLoader(dataset=X_train2, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(dataset=X_test2, batch_size=batch_size, shuffle=False)
X_train1 = torch.tensor(X_train1)
Y_train1 = torch.tensor(Y_train1)
X_test1 = torch.tensor(X_test1)
Y_test1 = torch.tensor(Y_test1)

# Initialize network
model = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size).to(device)

# Compile
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train
for epoch in range(num_epoch):
    data = X_train1.float().to(device=device)
    targets = Y_train1.float().to(device=device)
    # print(data.shape)
    # print(targets.shape)
    # data = data.reshape(data.shape[0], -1)
    pred = model(data)
    loss = criterion(pred, targets)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()


plt.plot(targets.tolist(), color='r')
plt.plot(pred.tolist(), color='b')
plt.show()
# # Save model and optimizer state

# torch.save(model.state_dict(), "LSTM.pytorch")
# torch.save(optimizer.state_dict(), "LSTM_optimizer.pytorch")


# # Initialize new model and optimizer
# test_model = LSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
# test_model = test_model.to(device=device)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # load saved states onto newly initialized model and optimizer
# model.load_state_dict(torch.load('LSTM.pytorch'))
# optimizer.load_state_dict(torch.load('LSTM_optimizer.pytorch'))
# model.eval()


# X_test1 = X_test1.float().to(device=device)
# # X_test1 = X_test1.reshape(X_test1.shape[0], -1)
# x = model(X_test1)
# y = Y_test1.float().to(device=device)
# y = y.reshape(y.shape[0], -1)
# # print(pred)
# # print(Y_test2.float.reshape(Y_test2.shape[0], -1))
# plt.plot(y.tolist(), color='r', label='real')
# plt.plot(x.tolist(), color='b', label='prediction')
# plt.legend()
# plt.show()