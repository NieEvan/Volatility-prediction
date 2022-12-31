# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 14:42:43 2022

@author: Administrator
"""

import numpy as np
import pandas
import pickle
from scipy.stats import norm
import scipy.optimize as opt
import yfinance as yf
import pandas as pd
import pandas.core
import datetime
import time
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

# from darts.models import NBEATSModel

"""Get data"""

data_feed = GetData()
df, X_train1, X_test1, Y_train1, Y_test1, scaler1 = data_feed.get_seq_data()

df2, X_train2, X_test2, Y_train2, Y_test2, scaler2 = data_feed.get_vec_data()





"""SVR"""
SVR_model = pickle.load(open('SVR_pickle', 'rb'))
# # Create different SVR models
# # svr_poly = SVR(kernel='poly', degree=2)
# svr_lin = SVR(kernel='linear')
# # svr_rbf = SVR(kernel='rbf')

# # Search for best hyperparameters
# para_grid = {'gamma': sp_rand(),
#           'C': sp_rand(),
#           'epsilon': sp_rand()}
# SVR_model = RandomizedSearchCV(svr_lin, para_grid)

# # Fit and train the model on the data
# print("fitting model")
# SVR_model.fit(X_train2, Y_train2) # the y is basically the realized volatility but shifted right by 1, so the x contains the current volatility and squared returns to predict the next timestep's volatility

# Prediction on the testing set
SVR_predictions = SVR_model.predict(X_test2)
SVR_predictions = pd.DataFrame(SVR_predictions)
SVR_predictions = SVR_predictions[:-1]

df2_test_copy = df2.copy()[-len(SVR_predictions):]
SVR_predictions.index = df2_test_copy.index
df2_test_copy['realized_vol'] = SVR_predictions[0]

SVR_predictions = scaler2.inverse_transform(df2_test_copy)
SVR_predictions = pd.DataFrame(SVR_predictions)

SVR_predictions.index = df2_test_copy.index

SVR_predictions = list(SVR_predictions[8])


# Calculate RMSE
rmse_svr = mse(Y_test2, SVR_predictions, squared=False) / 100
print('The RMSE value of SVR with Linear Kernel is {:.6f}'.format(rmse_svr))

# Plot out the prediction on top of the real value
plt.figure(figsize=(10, 6))
plt.plot(df2.index, [a/100 for a in df2['realized_vol']], label='Realized Volatility')
plt.plot(df2_test_copy.index, [b/100 for b in SVR_predictions], label='Volatility Prediction-SVR-GARCH')
plt.title('Volatility Prediction with SVR-GARCH (Linear)', fontsize=12)
plt.legend()
plt.show()

# # Save the model
# with open('SVR_pickle', 'wb') as f:
#     pickle.dump(SVR_model, f)








# """MLP"""
# from sklearn.neural_network import MLPRegressor

# NN_vol = MLPRegressor(learning_rate_init=0.001, random_state=1)
# para_grid_NN = {'hidden_layer_sizes': [(100, 50), (50, 50), (10, 100)],
#                 'max_iter': [500, 1000],
#                 'alpha': [0.00005, 0.0005 ]}
# MLP_model = RandomizedSearchCV(NN_vol, para_grid_NN)
# MLP_model.fit(X_train2, Y_train2)
# MLP_predictions = MLP_model.predict(X_test2)
# MLP_predictions = pd.DataFrame(MLP_predictions)
# MLP_predictions = MLP_predictions[:-1]
# MLP_predictions.index = df2[-len(Y_test2):].index

# rmse_MLP = mse(Y_test2, MLP_predictions, squared=False) / 100
# print('The RMSE value of NN is {:.6f}'.format(rmse_MLP))


# # rescaling back
# df_MLP_test_copy = df2.copy()[-len(MLP_predictions):]
# MLP_predictions.index = df_MLP_test_copy.index
# df_MLP_test_copy['realized_vol'] = MLP_predictions[0]

# MLP_predictions = scaler2.inverse_transform(df_MLP_test_copy)
# MLP_predictions = pd.DataFrame(MLP_predictions)
# MLP_predictions.index = df2[-len(Y_test2):].index

# MLP_predictions = list(MLP_predictions[8])

# plt.figure(figsize=(10, 6))
# plt.plot(df2['realized_vol'] / 100, label='Realized Volatility')
# plt.plot(df2[-len(Y_test2):].index, [a/100 for a in MLP_predictions], label='Volatility Prediction-MLP')
# plt.title('Volatility Prediction with MLP', fontsize=12)
# plt.legend()
# plt.show()

# with open('MLP_pickle', 'wb') as f:
#     pickle.dump(MLP_model, f)




 

"""Neural Network"""
NN_model = load_model("NN_model.h5")
# NN_model = Sequential()
# NN_model.add(Dense(128, input_dim=X_train2.shape[1])) #12
# NN_model.add(LeakyReLU(alpha=0.05))
# NN_model.add(Dropout(0.2))
# NN_model.add(Dense(64))
# NN_model.add(LeakyReLU(alpha=0.05))
# NN_model.add(Dropout(0.2))
# NN_model.add(Dense(64))
# NN_model.add(LeakyReLU(alpha=0.05))
# NN_model.add(Dropout(0.2))
# NN_model.add(Dense(32))
# NN_model.add(LeakyReLU(alpha=0.05))
# NN_model.add(Dense(1))

nadam = Nadam()

# NN_model.compile(loss='mean_squared_error', optimizer=nadam, metrics = ['acc'])
# print(NN_model.summary())

# NN_model.fit(np.array(X_train2), np.array(Y_train2), verbose=2, epochs=25)

NN_predictions = NN_model.predict(X_test2)
NN_predictions = pd.DataFrame(NN_predictions)
NN_predictions = NN_predictions[:-1]
NN_predictions.index = df2[-len(Y_test2):].index


rmse_NN = mse(Y_test2, NN_predictions, squared=False) / 100
print('The RMSE value of NN is {:.6f}'.format(rmse_NN))

# rescaling back
df_NN_test_copy = df2.copy()[-len(NN_predictions):]
NN_predictions.index = df_NN_test_copy.index
df_NN_test_copy['realized_vol'] = NN_predictions[0]

NN_predictions = scaler2.inverse_transform(df_NN_test_copy)
NN_predictions = pd.DataFrame(NN_predictions)

NN_predictions = list(NN_predictions[8])



plt.figure(figsize=(10, 6))
plt.plot(df2['realized_vol'] / 100, label='Realized Volatility')
plt.plot(df_NN_test_copy.index, [b/100 for b in NN_predictions], label='Volatility Prediction-NN')
plt.title('Volatility Prediction with Neural Network', fontsize=12)
plt.legend()
plt.show()

# NN_model.save('NN_model.h5')



"""XGBoost"""
# XGB_model = XGBRFRegressor(learning_rate=1.1)

# XGB_model.fit(X_train2, Y_train2)
XGB_model = pickle.load(open('XGB_pickle', 'rb'))

XGB_predictions = XGB_model.predict(X_test2)
XGB_predictions = pd.DataFrame(XGB_predictions)
XGB_predictions = XGB_predictions[:-1]
XGB_predictions.index = df2[-len(Y_test2):].index

rmse_MLP = mse(Y_test2, XGB_predictions, squared=False) / 100
print('The RMSE value of XGBoost is {:.6f}'.format(rmse_MLP))

df_XGB_test_copy = df2.copy()[-len(XGB_predictions):]
XGB_predictions.index = df_XGB_test_copy.index
df_XGB_test_copy['realized_vol'] = XGB_predictions[0]

XGB_predictions = scaler2.inverse_transform(df_XGB_test_copy)
XGB_predictions = pd.DataFrame(XGB_predictions)
XGB_predictions.index = df2[-len(Y_test2):].index

XGB_predictions = list(XGB_predictions[8])

plt.figure(figsize=(10, 6))
plt.plot(df2['realized_vol'] / 100, label='Realized Volatility')
plt.plot(df2[-len(Y_test2):].index, [a/100 for a in XGB_predictions], label='Volatility Prediction-XGBoost')
plt.title('Volatility Prediction with XGBoost', fontsize=12)
plt.legend()
plt.show()

# with open('XGB_pickle', 'wb') as f:
#     pickle.dump(XGB_model, f)






"""LSTM"""
LSTM_model = load_model('LSTM_model.h5')

# LSTM_model = keras.Sequential([
#     LSTM(128, input_shape=(X_train1.shape[1], X_train1.shape[2]), return_sequences=True),
#     Dropout(0.2),
#     LSTM(128, return_sequences=True),
#     Dropout(0.2),
#     LSTM(128, return_sequences=False),
#     Dense(64, activation='relu'),
#     Dense(1)
# ])


# history = LSTM_model.compile(optimizer=nadam, loss='mse', metrics=['accuracy'])
# LSTM_model.summary()
# LSTM_model.fit(X_train1, Y_train1, epochs=10, verbose=2)
LSTM_predictions = LSTM_model.predict(X_test1)


LSTM_predictions = pd.DataFrame(LSTM_predictions)
# LSTM_predictions = LSTM_predictions[:-1]
rmse_LSTM = mse(Y_test1, LSTM_predictions, squared=False) / 100
print('The RMSE value of LSTM is {:.6f}'.format(rmse_LSTM))


# rescaling back
df_LSTM_test_copy = df.copy()[-len(LSTM_predictions):]
LSTM_predictions.index = df_LSTM_test_copy.index
df_LSTM_test_copy['realized_vol'] = LSTM_predictions[0]
LSTM_predictions = scaler1.inverse_transform(df_LSTM_test_copy)
LSTM_predictions = pd.DataFrame(LSTM_predictions)
LSTM_predictions = list(LSTM_predictions[8])


plt.figure(figsize=(10, 6))
plt.plot(df['realized_vol'] / 100, label='Realized Volatility')
plt.plot(df_LSTM_test_copy.index, [a/100 for a in LSTM_predictions], label='Volatility Prediction-LSTM')
plt.title('Volatility Prediction with LSTM', fontsize=12)
plt.legend()
plt.show()

# LSTM_model.save('LSTM_model.h5')




# """Convolutional LSTM"""





"""GRU"""
GRU_model = load_model('GRU_model.h5')
# GRU_model = keras.Sequential([
#     GRU(128, input_shape=(X_train1.shape[1], X_train1.shape[2]), return_sequences=True),
#     Dropout(0.2),
#     # LSTM(128, return_sequences=True),
#     # Dropout(0.1),
#     Dense(128, activation='selu'),
#     GRU(128, return_sequences=False),
#     Dense(64, activation='selu'),
#     Dense(1)
# ])


# history = GRU_model.compile(optimizer=nadam, loss='mse', metrics=['accuracy'])
# GRU_model.summary()
# GRU_model.fit(X_train1, Y_train1, epochs=10, verbose=2)
GRU_predictions = GRU_model.predict(X_test1)


GRU_predictions = pd.DataFrame(GRU_predictions)
# LSTM_predictions = LSTM_predictions[:-1]
rmse_GRU = mse(Y_test1, GRU_predictions, squared=False) / 100
print('The RMSE value of GRU is {:.6f}'.format(rmse_GRU))

# rescaling back
df_GRU_test_copy = df.copy()[-len(GRU_predictions):]
GRU_predictions.index = df_GRU_test_copy.index
df_GRU_test_copy['realized_vol'] = GRU_predictions[0]
GRU_predictions = scaler1.inverse_transform(df_GRU_test_copy)
GRU_predictions = pd.DataFrame(GRU_predictions)
GRU_predictions = list(GRU_predictions[8])


plt.figure(figsize=(10, 6))
plt.plot(df['realized_vol'] / 100, label='Realized Volatility')
plt.plot(df_GRU_test_copy.index, [a/100 for a in GRU_predictions], label='Volatility Prediction-GRU')
plt.title('Volatility Prediction with GRU', fontsize=12)
plt.legend()
plt.show()

# with open('GRU_pickle', 'wb') as f:
#     pickle.dump(GRU_model, f)






# # """NBEATSModel"""
# # NBEATS_model = NBEATSModel(chunk_length=15, output_chunk_length=1, n_epochs=100, random_state=0)
# # NBEATS_model.fit(X_train2, val_series=X_test2)




    
"""Meta model"""
# meta_model = pickle.load(open('metamodel_pickle', 'rb'))
meta_input = df2[-len(NN_predictions):].copy()
meta_input = meta_input.drop(['realized_vol'], axis=1)
meta_input['SVR'] = SVR_predictions
# meta_input['MLP'] = MLP_predictions
meta_input['NN'] = NN_predictions
meta_input['XGBoost'] = XGB_predictions
meta_input = meta_input[-len(LSTM_predictions):]
meta_input['LSTM'] = LSTM_predictions
# meta_input['LSTM'] = meta_input['LSTM'].shift(len(NN_predictions)-len(LSTM_predictions))
meta_input['last_realized_vol'] = list(df2['realized_vol'][-(len(LSTM_predictions)+1):-1])
meta_input['realized_vol'] = list(df2['realized_vol'][-len(LSTM_predictions):])
# meta_input = meta_input[-len(LSTM_predictions):]




# Scaling the data
scaler3 = preprocessing.StandardScaler()
scaled_meta_input = scaler3.fit_transform(meta_input)
print(meta_input)

X = scaled_meta_input[:, 0:13]
Y = scaled_meta_input[:, 13]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=False)


meta_model = XGBRFRegressor()

meta_model.fit(X_train, Y_train)

pred = meta_model.predict(X_test)
pred = pd.DataFrame(pred)


# pred.index = df2[-len(Y_test):].index

rmse_meta = mse(Y_test, pred, squared=False) / 100
print('The RMSE value of meta model is {:.6f}'.format(rmse_meta))


df_meta_test_copy = meta_input.copy()[-len(pred):]
pred.index = df_meta_test_copy.index
df_meta_test_copy['realized_vol'] = pred[0]
print(df_meta_test_copy)
pred = scaler3.inverse_transform(df_meta_test_copy)
pred = pd.DataFrame(pred)

pred = list(pred[13])


plt.plot(df['realized_vol'][-len(pred):] / 100, color='b', label='Realized Volatility')
plt.plot(df['realized_vol'][-len(pred):].index, [a/100 for a in pred], color='r', label='Meta Model Predictions')
plt.legend()
plt.show()


with open("metamodel_pickle", 'wb') as f:
    pickle.dump(meta_model, f)

# df_meta_test_copy = meta_input.copy()[-len(pred):]
# pred.index = df_meta_test_copy.index
# df_meta_test_copy['realized_vol'] = pred[0]

# pred = scaler3.inverse_transform(df_meta_test_copy)
# pred = pd.DataFrame(pred)

# pred = list(pred.iloc[:, 2])


