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
from get_data3 import GetData
from xgboost import XGBRFRegressor
import statistics

# from darts.models import NBEATSModel

"""Get data"""
data = GetData()
df, X_train1, X_test1, Y_train1, Y_test1, scaler1, WINDOWSIZE = data.get_seq_data()
df2, X_train2, X_test2, Y_train2, Y_test2, scaler2 = data.get_vec_data()


"""Models"""
SVR_model = pickle.load(open('SVR_pickle', 'rb'))
SVR_predictions = data.vec_test(SVR_model, X_test2) # returns unscaled list
 
NN_model = load_model("NN_model.h5")
NN_predictions = data.vec_test(NN_model, X_test2)

XGB_model = XGBRFRegressor(learning_rate=1.1)
XGB_model.fit(X_train2, Y_train2)
XGB_predictions = data.vec_test(XGB_model, X_test2)

LSTM_model = load_model('LSTM_model.h5') # best lstm model yet
LSTM_predictions = data.seq_test(LSTM_model, X_test1)

GRU_model = load_model('GRU_model.h5')
GRU_predictions = data.seq_test(GRU_model, X_test1)

# # """NBEATSModel"""
# # NBEATS_model = NBEATSModel(chunk_length=15, output_chunk_length=1, n_epochs=100, random_state=0)
# # NBEATS_model.fit(X_train2, val_series=X_test2)
    
"""Meta model"""
# meta_model = pickle.load(open('metamodel_pickle', 'rb'))
meta_input = pd.DataFrame()
meta_input['SVR'] = pd.Series(SVR_predictions)
# meta_input['MLP'] = MLP_predictions
# meta_input['NN'] = pd.Series(NN_predictions)
meta_input['XGBoost'] = pd.Series(XGB_predictions)
meta_input['LSTM'] = pd.Series(LSTM_predictions)
meta_input['GRU'] = pd.Series(GRU_predictions)
meta_input['GRU'] = meta_input['GRU'].shift(len(NN_predictions)-len(GRU_predictions))
meta_input['LSTM'] = meta_input['LSTM'].shift(len(NN_predictions)-len(LSTM_predictions))
meta_input['realized_vol'] = list(df['realized_vol'][-(len(NN_predictions)):])
meta_input['next_day_vol'] = list(df['next_day_vol'][-len(NN_predictions):])
meta_input = meta_input[-len(LSTM_predictions):]


# Scaling the data
scaler3 = preprocessing.StandardScaler()
scaled_meta_input = scaler3.fit_transform(meta_input)
print(meta_input)

# Getting train and test data for meta model
X = scaled_meta_input[:, 0:5]
Y = scaled_meta_input[:, 5]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=False)

# Train and predict using meta model
meta_model = XGBRFRegressor()
meta_model.fit(X_train, Y_train)
pred = meta_model.predict(X_test)
pred = pd.DataFrame(pred)

# Rescale the meta model predictions back to normal
df_meta_test_copy = meta_input.copy()[-len(pred):]
pred.index = df_meta_test_copy.index
df_meta_test_copy['next_day_vol'] = pred[0]
pred = scaler3.inverse_transform(df_meta_test_copy)
pred = pd.DataFrame(pred)
pred = list(pred[5])

# Calculate difference between true values and predictions of meta model
diff = list()
for item1, item2 in zip(df['next_day_vol'][-len(pred):], pred):
    item = item1 - item2
    diff.append(item)

# Plot lineplot, scatter plot, and distribution
figure, axis = plt.subplots(3, 1)
axis[0].plot(df['next_day_vol'][-len(pred):] / 100, color='b', label='Realized Volatility')
axis[0].plot(df['next_day_vol'][-len(pred):].index, [a/100 for a in pred], color='r', label='Meta Model Predictions')
axis[0].set_title("Lineplot：Prediction vs real values")
axis[0].legend()

axis[1].scatter([a/100 for a in pred], df['next_day_vol'][-len(pred):] / 100)
axis[1].set_title("Scatter plot: Prediction vs real values")

axis[2].hist([a/100 for a in diff], bins=142)
axis[2].set_title("Distribution of predictions")

figure.tight_layout()
plt.show()


# Calculate RMSE of each model on the same data length as the meta model's input
rmse_NN_meta = mse(df['next_day_vol'][-len(pred):], NN_predictions[-len(pred):], squared=False) / 100
rmse_SVR_meta = mse(df['next_day_vol'][-len(pred):], SVR_predictions[-len(pred):], squared=False) / 100
rmse_LSTM_meta = mse(df['next_day_vol'][-len(pred):], LSTM_predictions[-len(pred):], squared=False) / 100
rmse_GRU_meta = mse(df['next_day_vol'][-len(pred):], GRU_predictions[-len(pred):], squared=False) / 100
rmse_XGBoost = mse(df['next_day_vol'][-len(pred):], XGB_predictions[-len(pred):], squared=False) / 100
rmse_meta = mse(df['next_day_vol'][-len(pred):], pred, squared=False) / 100

# # plt.plot(df['realized_vol'][-len(pred):] / 100, color='b', label='Realized Volatility')
# # plt.plot(df['realized_vol'][-len(pred):].index, [a/100 for a in pred], color='r', label='Meta Model Predictions')
# plt.plot(df['realized_vol'][-len(pred):] / 100, color='b', label='Realized Volatility')
# plt.plot(df_GRU_test_copy[-len(pred):].index, [a/100 for a in GRU_predictions[-len(pred):]], color='y', label='GRU Predictions')
# plt.legend()
# plt.show()

# plt.plot(df['realized_vol'][-len(pred):] / 100, color='b', label='Realized Volatility')
# plt.plot(df_LSTM_test_copy[-len(pred):].index, [a/100 for a in LSTM_predictions[-len(pred):]], color='g', label='LSTM Predictions')
# plt.legend()
# plt.show()

# plt.plot(df['realized_vol'][-len(pred):] / 100, color='b', label='Realized Volatility')
# plt.plot(df2_test_copy[-len(pred):].index, [a/100 for a in SVR_predictions[-len(pred):]], color='orange', label='SVR Predictions')
# plt.legend()
# plt.show()

# plt.plot(df['realized_vol'][-len(pred):] / 100, color='b', label='Realized Volatility')
# plt.plot(df_NN_test_copy[-len(pred):].index, [a/100 for a in NN_predictions[-len(pred):]], color='purple', label='NN Predictions')
# plt.legend()
# plt.show()

# Output
print(rmse_meta, rmse_NN_meta, rmse_SVR_meta, rmse_LSTM_meta, rmse_XGBoost)
print('The RMSE value of meta model is {:.6f}'.format(rmse_meta))
print("Mean of distribution of (difference between predictions and real values is % s"%(statistics.mean(diff))) 
print("Standard Deviation of the difference between predictions and real values is % s "%(statistics.stdev(diff)))




