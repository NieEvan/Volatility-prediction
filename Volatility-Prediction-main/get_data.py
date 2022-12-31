from dotenv import load_dotenv, find_dotenv
import os
import pprint
from pymongo import MongoClient
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# import talib
import requests
import io
from datetime import datetime, timedelta
from bson.objectid import ObjectId
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse


# load_dotenv(find_dotenv())
# password = os.environ.get("MONGODB_PWD")

# connection_string = f"mongodb+srv://evanNie:{password}@volatilitypredictiondat.lrj2cnx.mongodb.net/?retryWrites=true&w=majority"
# client = MongoClient(connection_string)

# db_names = client.list_database_names()
# vol_db = client.volatility_prediction
# collections = vol_db.list_collection_names()


# def live(symbol):
#     collection = vol_db[symbol]
    
#     one_min = collection.find_one({'interval':'1min'})
#     one_min_df = pd.DataFrame(one_min)
#     # one_min_df.drop(columns=['_id', 'interval'], inplace=True)
    
#     five_min = collection.find_one({'interval':'5min'})
#     five_min_df = pd.DataFrame(five_min)
#     # five_min_df.drop(columns=['_id', 'interval'], inplace=True)
    
#     thirty_min = collection.find_one({'interval':'5min'})
#     thirty_min_df = pd.DataFrame(thirty_min)
#     # thirty_min_df.drop(columns=['_id', 'interval'], inplace=True)
    
#     one_hour = collection.find_one({'interval':'1d'})
#     one_hour_df = pd.DataFrame(one_hour)
#     # one_hour_df.drop(columns=['_id', 'interval'], inplace=True)
    
#     one_day = collection.find_one({'interval':'1d'})
#     one_day_df = pd.DataFrame(one_day)
#     # one_day_df.drop(columns=['_id', 'interval'], inplace=True)
    
#     return one_min_df, five_min_df, thirty_min_df, one_hour_df, one_day_df


# def pre_proc(data):
#     data['ret'] = 100 * (data['Adj Close'].pct_change()[1:])
#     # Calculate squared returns
#     data['returns_svm'] = data['ret'] ** 2
#     # Calculate realized volatility
#     data['realized_vol'] = data['ret'].rolling(5).std()
#     # Remove the first 5 rows due to computing the 5 day volatility
#     df = data[5:].copy()
#     df = df.reset_index(drop=True)
#     df = df[:-1]
#     print(df)
#     return df
        
        

class GetData():
    def __init__(self):
        self.data = pd.read_csv('AAPL.csv')


    def get_seq_data(self):
        """Sequence data (For LSTM and RNN)"""
        data1 = self.data.copy()
        # Calculate indicators
        # data1['ATR'] = talib.ATR(data1['High'], data1['Low'], data1['Close'], timeperiod=14)
        # data1['OBV'] = talib.OBV(data1['Close'], data1['Volume'])

        # Calculate returns
        data1['ret'] = 100 * (data1['Adj Close'].pct_change()[1:])
        # Calculate squared returns
        data1['returns_svm'] = data1['ret'] ** 2
        # Calculate realized volatility
        data1['realized_vol'] = data1['ret'].rolling(5).std()
        # print(data1)
        # Remove the first 5 rows due to computing the 5 day volatility
        self.df = data1[5:].copy()
        self.df = self.df.reset_index()
        self.df.index = pd.to_datetime(self.df['Date'], format='%Y.%m.%d')
        self.df.pop('Date')
        self.df.pop('index')
        self.df = self.df[:-1]
        # print(self.df)
        self.scaler1 = preprocessing.StandardScaler()
        df_scaled = self.scaler1.fit_transform(self.df)

        # print(df)
        X = []
        Y = []
        
        def df_to_XY(df, window_size=15, predict_size=1):
          for i in range(window_size, len(df) - predict_size + 1):
              X.append(df[i - window_size:i, 0:df.shape[1]])
              Y.append(df[i + predict_size - 1:i + predict_size, 8]) # 8 = realized_vol column
          return np.array(X), np.array(Y)
        # print(df_scaled[:, 7])

        WINDOW_SIZE = 15
        predict_len = 1
        X1, Y1 = df_to_XY(df_scaled, WINDOW_SIZE, predict_len)

        self.X_train1, X_test1, Y_train1, self.Y_test1 = train_test_split(X1, Y1, test_size=0.3, shuffle=False)

        # print(df.shape)
        # print(df_scaled.shape)
        # print(X_train1.shape)
        # print(Y_train1.shape)
        return self.df, self.X_train1, X_test1, Y_train1, self.Y_test1, self.scaler1, WINDOW_SIZE


    def get_vec_data(self):
        """Vector to vector data (For most models)"""
        self.df2 = self.df.copy()

        def list_of_lists(lst):
            return [[el] for el in lst]

        # Get the training and testing X's
        self.scaler2 = preprocessing.StandardScaler()
        df2_scaled = self.scaler2.fit_transform(self.df2)

        Y2 = np.roll(df2_scaled[:, 8], -1)  # target vairable future 1 day volatility

        Y2 = Y2[:-1]

        Y2 = list_of_lists(Y2)
        # print(Y2)
        X_train2 = df2_scaled[:len(self.X_train1)]
        X_test2 = df2_scaled[len(self.X_train1):]
        Y_train2 = Y2[:len(self.X_train1)]
        self.Y_test2 = Y2[len(self.X_train1):]

        return self.df2, X_train2, X_test2, Y_train2, self.Y_test2, self.scaler2


    def vec_test(self, model, data):
        predictions = model.predict(data)
        predictions = pd.DataFrame(predictions)
        print(predictions)
        predictions = predictions[:-1] # compromise for not having the real volatility of one step into the future

        # Calculate RMSE
        rmse = mse(self.Y_test2, predictions, squared=False) / 100
        print('The RMSE value of model with model is {:.6f}'.format(rmse))


        test_copy = self.df2.copy()[-len(predictions):]
        predictions.index = test_copy.index
        test_copy['realized_vol'] = predictions[0]

        predictions = self.scaler2.inverse_transform(test_copy)

        predictions = pd.DataFrame(predictions)
        predictions.index = test_copy.index
        predictions = list(predictions[8]) # get te predicted volatility
        
        # Plot out the prediction on top of the real value
        plt.figure(figsize=(10, 6))
        plt.plot(self.df2.index, [a/100 for a in self.df2['realized_vol']], label='Realized Volatility')
        plt.plot(self.df2.copy()[-len(predictions)-1:-1].index, [b/100 for b in predictions], label='Volatility Prediction')
        plt.title('Volatility Prediction with model on test set', fontsize=12)
        plt.legend()
        plt.show()

        return predictions # return unscaled predictions


    def seq_test(self, model, data):
        predictions = model.predict(data)
        predictions = pd.DataFrame(predictions)

        rmse = mse(self.Y_test1, predictions, squared=False) / 100
        print('The RMSE value of model with Linear Kernel is {:.6f}'.format(rmse))


        test_copy = self.df.copy()[-len(predictions):]
        predictions.index = test_copy.index
        test_copy['realized_vol'] = predictions[0]
        predictions = self.scaler1.inverse_transform(test_copy)
        predictions = pd.DataFrame(predictions)
        predictions = list(predictions[8])
        # Plot out the prediction on top of the real value
        plt.figure(figsize=(10, 6))
        plt.plot(self.df2['realized_vol'] / 100, label='Realized Volatility')
        plt.plot(test_copy.index, [b/100 for b in predictions], label='Volatility Prediction')
        plt.title('Volatility Prediction with model on test set', fontsize=12)
        plt.legend()
        plt.show()

        return predictions # return unscaled predictions
        
        