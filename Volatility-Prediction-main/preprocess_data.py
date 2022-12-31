import pandas as pd
import numpy as np
from sklearn import preprocessing


def pre_proc_df(df):
    df['ret'] = 100 * (df['Adj Close'].pct_change()[1:])
    # Calculate realized volatility
    df['realized_vol'] = df['ret'].rolling(5).std()
    # Calculate squared returns
    df['returns_svm'] = df['ret'] ** 2
    
    # Remove the first 5 rows due to computing the 5 day volatility
    df2 = df[5:].copy()
    df2 = df2.reset_index()
    df2.index = pd.to_datetime(df2['Date'], format='%Y.%m.%d')
    df2.pop('Date')
    df2.pop('index')
    
    pre_proc_df.scaler = preprocessing.StandardScaler()
    scaled_inputs = pre_proc_df.scaler.fit_transform(df2)
    
    return df2, scaled_inputs
    

def rescale_pred(pred:np.array, df):
    
    pred = pd.DataFrame(pred)

    df_test_copy = df.copy()[-len(pred):]
    pred.index = df_test_copy.index
    df_test_copy['realized_vol'] = pred[0]

    pred = pre_proc_df.scaler.inverse_transform(df_test_copy)
    pred = pd.DataFrame(pred)

    pred.index = df_test_copy.index
    pred = list(pred[7])
    
    return pred





"""For meta model"""

def scale(inputs):
    scale.scaler = preprocessing.RobustScaler()
    scaled_inputs = scale.scaler.fit_transform(inputs)
    
    return scaled_inputs
    

def predict(model, scaled_inputs, df):
    # predict
    pred = model.predict(scaled_inputs)
    pred = pd.DataFrame(pred)
    df_pred = df.copy()
    pred.index = df_pred.index
    df_pred['realized_vol'] = pred[0]
    
    return df_pred


def rescale(inputs):
    pred = scale.scaler.inverse_transform(inputs)
    pred = pd.DataFrame(pred)
    pred = list(pred[2])
    
    return pred