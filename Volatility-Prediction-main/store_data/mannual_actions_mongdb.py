from dotenv import load_dotenv, find_dotenv
import os
import pprint
from pymongo import MongoClient
import pandas as pd
import numpy as np
import requests
import io
from datetime import datetime, timedelta
 

load_dotenv(find_dotenv())
password = os.environ.get("MONGODB_PWD")

connection_string = f"mongodb+srv://evanNie:{password}@volatilitypredictiondat.lrj2cnx.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(connection_string)

db_names = client.list_database_names()
vol_db = client.volatility_prediction
collections = vol_db.list_collection_names()

# # delete every document in every collection in the database
# for collection in collections:
#     vol_db[f'{collection}'].delete_many({})

# delete every collection in the database
for collection in collections:
    vol_db[f'{collection}'].drop()