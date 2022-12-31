from dotenv import load_dotenv, find_dotenv
import os
import pprint
from pymongo import MongoClient
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import talib
import requests
import io
from datetime import datetime, timedelta
from bson.objectid import ObjectId


pd.set_option('display.max_columns', None)


load_dotenv(find_dotenv())
password = os.environ.get("MONGODB_PWD")

connection_string = f"mongodb+srv://evanNie:{password}@volatilitypredictiondat.lrj2cnx.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(connection_string)

db_names = client.list_database_names()
vol_db = client.volatility_prediction
collections = vol_db.list_collection_names()
print(collections)


now = datetime.now()
table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
tickers = table[0]['Symbol'].to_list()

def store(data, symbol):
    collections = vol_db[symbol]
    collections.insert_one(data)


def pre_proc(data):
    data['ret'] = 100 * (data['Adj Close'].pct_change()[1:])
    # Calculate squared returns
    data['returns_svm'] = data['ret'] ** 2
    # Calculate realized volatility
    data['realized_vol'] = data['ret'].rolling(5).std()
    # Remove the first 5 rows due to computing the 5 day volatility
    df = data.copy()
    df.reset_index(inplace=True)
    # print(df)
    df = df.rename(columns={'index':'Datetime'})
    df['Datetime'] = df['Datetime'].dt.strftime("%m-%d-%Y, %H:%M:%S")
    # print(df)
    return df


def main():
    for ticker in tickers:

# Get data
        # one_min_data = yf.download(ticker, start=now - timedelta(days=7),
        #                     end=now, interval="1m")
        five_min_data = yf.download(ticker, start =now - timedelta(days=21),
                            end=now, interval="5m")
        fifteen_min_data = yf.download(ticker, start =now - timedelta(days=21),
                            end=now, interval="15m")
        thirty_min_data = yf.download(ticker, start=now - timedelta(days=21),
                            end=now, interval="30m")
        one_hour_data = yf.download(ticker, start=now - timedelta(days=21),
                            end=now, interval="1h")
        # one_day_data = yf.download(ticker, start=now - timedelta(days=7),
        #                 end=now, interval="1d")
        print(f'finished download data for {ticker}')
        
        if len(five_min_data) == 0 or len(thirty_min_data) == 0 or len(fifteen_min_data) == 0 or len(one_hour_data) == 0:
            print(f'cannot retrieve data of symbol: {ticker}')
            tickers.remove(ticker)
            continue
        
        # vol_db.create_collection(f'{ticker}')

        min5_proc = pre_proc(five_min_data)
        min15_proc = pre_proc(fifteen_min_data)
        min30_proc = pre_proc(thirty_min_data)
        h1_proc = pre_proc(one_hour_data)
        # d1_proc = pre_proc(one_day_data)
        
        
        # min5_proc = min5_proc[60:]
        # min15_proc = min15_proc[20:]
        # min30_proc = min30_proc[10:]
        # h1_proc = h1_proc[5:]
        # h1_proc = h1_proc[:-1]
        # min30_proc = min30_proc[:-1]
        # min15_proc = min15_proc[:-2]
        # min5_proc = min5_proc[:-6]
        # min5_proc.reset_index(inplace=True, drop=True)
        # min15_proc.reset_index(inplace=True, drop=True)
        # min30_proc.reset_index(inplace=True, drop=True)
        h1_proc.reset_index(inplace=True, drop=True)
        # Add prefix to column names of each timeframe's dataframe
        min5_proc = min5_proc.add_prefix('5min_')
        min15_proc = min15_proc.add_prefix('15min_')
        min30_proc = min30_proc.add_prefix('30min_')
        h1_proc = h1_proc.add_prefix('1h_')
        
        
        # print(h1_proc)
        # print(min30_proc)
        # print(min15_proc)
        # print(min5_proc)

        
        """Create repeating dataframes for different timeframes"""
        
        # 30min
        ### concat the 1h rows onto the 30min dataframe
        min30_proc_df = pd.DataFrame()
        min30_proc_df = pd.concat([min30_proc_df, h1_proc.head(1)])
        for i in range(1, len(min30_proc)):
            if min30_proc['30min_Datetime'][i] in h1_proc['1h_Datetime'].values: # if the current time is present in the 1h time index, then concat the row with the corresponding row in the 1h dataframe
                min30_proc_df = pd.concat([min30_proc_df, h1_proc.loc[h1_proc['1h_Datetime'] == min30_proc['30min_Datetime'][i]]])
            else: # if the current time is in between the hourly data of h1_proc, then append the corresponding row in h1_proc to the last iteration's index
                min30_proc_df = pd.concat([min30_proc_df, h1_proc.loc[h1_proc['1h_Datetime'] == min30_proc['30min_Datetime'][i-1]]])
        min30_proc_df.reset_index(inplace=True, drop=True)
        min30_proc_df_final = pd.concat([min30_proc, min30_proc_df], axis=1)

        
        
        
        # 15min
        min15_proc_df = pd.DataFrame()
        min15_proc_df = pd.concat([min15_proc_df, min30_proc_df_final.head(1)])
        for i in range(1, len(min15_proc)):
            if min15_proc['15min_Datetime'][i] in min30_proc_df_final['30min_Datetime'].values: # if the current time is present in the 1h time index, then concat the row with the corresponding row in the 1h dataframe
                min15_proc_df = pd.concat([min15_proc_df, min30_proc_df_final.loc[min30_proc_df_final['30min_Datetime'] == min15_proc['15min_Datetime'][i]]])
            else: # if the current time is in between the hourly data of h1_proc, then append the corresponding row in h1_proc to the last iteration's index
                min15_proc_df = pd.concat([min15_proc_df, min30_proc_df_final.loc[min30_proc_df_final['30min_Datetime'] == min15_proc['15min_Datetime'][i-1]]])
        min15_proc_df.reset_index(inplace=True, drop=True)
        min15_proc_df_final = pd.concat([min15_proc, min15_proc_df], axis=1)
        
        
        # 5min
        min5_proc_df = pd.DataFrame()
        
        min5_proc_df = pd.concat([min5_proc_df, min15_proc_df_final.head(1)])
        if min5_proc['5min_Datetime'][1] not in min15_proc_df_final['15min_Datetime'].values:
            min5_proc_df = pd.concat([min5_proc_df, min15_proc_df_final.head(1)])
        for i in range(2, len(min5_proc)):
            if min5_proc['5min_Datetime'][i] in min15_proc_df_final['15min_Datetime'].values: # if the current time is present in the 1h time index, then concat the row with the corresponding row in the 1h dataframe
                min5_proc_df = pd.concat([min5_proc_df, min15_proc_df_final.loc[min15_proc_df_final['15min_Datetime'] == min5_proc['5min_Datetime'][i]]])
            else: # if the current time is in between the hourly data of h1_proc, then append the corresponding row in h1_proc to the last iteration's index
                if min5_proc['5min_Datetime'][i-1] in min15_proc_df_final['15min_Datetime'].values:
                    min5_proc_df = pd.concat([min5_proc_df, min15_proc_df_final.loc[min15_proc_df_final['15min_Datetime'] == min5_proc['5min_Datetime'][i-1]]])
                elif min5_proc['5min_Datetime'][i-2] in min15_proc_df_final['15min_Datetime'].values:
                    min5_proc_df = pd.concat([min5_proc_df, min15_proc_df_final.loc[min15_proc_df_final['15min_Datetime'] == min5_proc['5min_Datetime'][i-2]]])
        min5_proc_df.reset_index(inplace=True, drop=True)    
        min5_proc_df_final = pd.concat([min5_proc, min5_proc_df], axis=1)
        # print(min5_proc_df_final['5min_Datetime'].head(50), min5_proc_df_final['15min_Datetime'].head(50), min5_proc_df_final['1h_Datetime'].head(50))
        # print(min5_proc_df_final.tail(60))

        min5_proc_df_final = min5_proc_df_final[54:]
        min5_proc_df_final = min5_proc_df_final[:-1]
        # date5min = min5_proc_df_final['5min_Datetime']
        # date15min = min5_proc_df_final['15min_Datetime']
        # date30min = min5_proc_df_final['30min_Datetime']
        # date1h = min5_proc_df_final['1h_Datetime']
        min5_proc_df_final = min5_proc_df_final.drop(columns=['1h_Datetime', '30min_Datetime', '15min_Datetime', '1h_Close', '30min_Close', '15min_Close', '5min_Close'])

        
        min5_proc_df_final = min5_proc_df_final.set_index('5min_Datetime')
        # print(min5_proc_df_final)


        # Store data
        # store(one_day_data, ticker)
        data = min5_proc_df_final.to_dict()
        store(data, ticker)

if __name__ == '__main__':
    main()