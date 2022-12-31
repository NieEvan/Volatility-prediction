from dotenv import load_dotenv, find_dotenv
import os
import pprint
from pymongo import MongoClient
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import talib
import requests
import io
from datetime import datetime, timedelta
from bson.objectid import ObjectId
import torch
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import torch.autograd as autograd
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm.notebook import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn as nn
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
        
        
def create_sequences(input_data: pd.DataFrame, target_column, sequence_length):
        sequences = []
        data_size = len(input_data) 
        for i in range(data_size - sequence_length):
            sequence = input_data[i:i+sequence_length]
            label = input_data.iloc[i + sequence_length][target_column]
            sequences.append((sequence, label))

        return sequences # returns a tuple: (features and labels up to the current time, next label)

"""GET DATA"""
data = pd.read_csv('AAPL.csv', parse_dates=['Date'])
# self.x = torch.from_numpy(self.data[:, :8])
# self.y = torch.from_numpy(self.data[:, 8])
# Calculate indicators
# data1['ATR'] = talib.ATR(data1['High'], data1['Low'], data1['Close'], timeperiod=14)
# data1['OBV'] = talib.OBV(data1['Close'], data1['Volume'])

# Calculate returns
data1 = data.copy()

data1['ret'] = 100 * (data1['Adj Close'].pct_change()[1:])
# Calculate squared returns
data1['returns_svm'] = data1['ret'] ** 2
# Calculate realized volatility
data1['realized_vol'] = data1['ret'].rolling(5).std()
# print(data1)
# Remove the first 5 rows due to computing the 5 day volatility
df = data1[5:].copy()
df = df.reset_index()
df.index = pd.to_datetime(df['Date'], format='%Y.%m.%d')
df.pop('Date')
df.pop('index')
df = df[:-1]

train_size = int(len(df)*0.8)
train_df, test_df = df[:train_size], df[train_size:]
# print(train_df)
scaler1 = MinMaxScaler(feature_range=(-1, 1))
df_scaled = scaler1.fit(df)

scaled_train_df = pd.DataFrame(
    scaler1.transform(train_df),
    index=train_df.index,
    columns=train_df.columns
)

scaled_test_df = pd.DataFrame(
    scaler1.transform(test_df),
    index=test_df.index,
    columns=test_df.columns
)


SEQUENCE_LENGTH = 60

train_sequences = create_sequences(scaled_train_df, "realized_vol", SEQUENCE_LENGTH)
test_sequences = create_sequences(scaled_test_df, "realized_vol", SEQUENCE_LENGTH)

# sample_data = pd.DataFrame(dict(
#     features=[1,2,3,4,5],
#     label=[6,7,8,9,10]
# ))

# sample_sequences = create_sequences(sample_data, "label", sequence_length=SEQUENCE_LENGTH)
# print(sample_sequences[0])
# print(sample_sequences[0][1])


class seqDataset(Dataset): # inherit from torch.utils.Dataset
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        return dict(
            sequence=torch.Tensor(sequence.to_numpy()),
            label=torch.tensor(label).float()
        )


class seqDataModule(pl.LightningDataModule):
    def __init__(self, train_sequences, test_sequences, batch_size=32):
        super().__init__()
        self.train_sequences = train_sequences
        self.test_sequences = test_sequences
        self.batch_size = batch_size
    
    def setup(self):
        self.train_dataset = seqDataset(self.train_sequences)
        self.test_dataset = seqDataset(self.test_sequences)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1
        )



epochs = 10
batch = 32

data_module = seqDataModule(train_sequences, test_sequences, batch_size=batch)
data_module.setup()

train_dataset = seqDataset(train_sequences)
# for item in train_dataset:
#     print(item)
#     print(item['sequence'].shape)
#     print(item['label'].shape)
#     print(item['label'])
#     break


class LSTM_model(nn.Module):
    def __init__(self, n_features, n_hidden=128, n_layers=3):
        super().__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            batch_first=True,
            num_layers=n_layers,
            dropout=0.2
        )

        self.fc = nn.Linear(n_hidden, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = hidden[-1]

        return self.fc(out)


class LSTM(pl.LightningDataModule):
    def __init__(self, n_features: int):
        super().__init__()
        self.model = LSTM_model(n_features)
        self.criterion = nn.MSELoss()

    def forward(self, x, labels=None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels.unsqueeze(dim=1))
        return loss, output
    
    def training_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        loss, outputs = self(sequences, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss}
    
    def test_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        loss, outputs = self(sequences, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.001)

print(scaled_train_df.shape[1])
# model = LSTM(n_features=scaled_train_df.shape[1])



        # return self.df, self.X_train1, X_test1, Y_train1, Y_test1, scaler1, WINDOW_SIZE

#     def get_vec_data(self):
#         """Vector to vector data (For most models)"""
#         df2 = self.df.copy()

#         def list_of_lists(lst):
#             return [[el] for el in lst]


#         # Get the training and testing X's
#         scaler2 = preprocessing.StandardScaler()
#         df2_scaled = scaler2.fit_transform(df2)

#         Y2 = np.roll(df2_scaled[:, 8], -1)  # target vairable future 1 day volatility

#         Y2 = Y2[:-1]

#         Y2 = list_of_lists(Y2)
#         # print(Y2)
#         X_train2 = df2_scaled[:len(self.X_train1)]
#         X_test2 = df2_scaled[len(self.X_train1):]
#         Y_train2 = Y2[:len(self.X_train1)]
#         Y_test2 = Y2[len(self.X_train1):]

#         return df2, X_train2, X_test2, Y_train2, Y_test2, scaler2



# # data = GetData()
# # df, X_train1, X_test1, Y_train1, Y_test1, scaler1 = data.get_seq_data()
# # print(df, df.shape)
# # print(X_test1.shape)
# # print(X_train1.shape)
# # print(Y_train1.shape)
