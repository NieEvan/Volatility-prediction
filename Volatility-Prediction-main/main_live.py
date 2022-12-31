import pandas as pd
import numpy as np
from sklearn import preprocessing
import pickle
import preprocess_data as pre
import io
import requests


url="https://pkgstore.datahub.io/core/nasdaq-listings/nasdaq-listed_csv/data/7665719fb51081ba0bd834fde71ce822/nasdaq-listed_csv.csv"
s = requests.get(url).content
companies = pd.read_csv(io.StringIO(s.decode('utf-8')))

symbols = companies['Symbol'].tolist()
print(len(symbols))
end()

SVR_model = pickle.load(open('SVR_pickle.pkl', 'rb'))
NN_model = pickle.load(open('NN_pickle.pkl', 'rb'))
MLP_model = pickle.load(open('MLP_pickle.pkl', 'rb'))
XGB_model = pickle.load(open('XGB_pickle.pkl', 'rb'))
meta_model = pickle.load(open('metamodel_pickle.pkl', 'rb'))




"""Get data"""
# Update data every hour

# Get data
data = pd.read()



def main(data): # data must include the past 5 data points and the current one
    df, df_scaled = pre.pre_proc_df(data)
    
    # predict using base models
    svr_pred = SVR_model.predict(data)
    nn_pred = NN_model.predict(data)
    mlp_pred = MLP_model.predict(data)
    xgb_pred = XGB_model.predict(data)
    
    # rescale base models' predictions
    svr_pred = pre.rescale_pred(svr_pred, df)
    nn_pred = pre.rescale_pred(nn_pred, df)
    mlp_pred = pre.rescale_pred(mlp_pred, df)
    
    
    # create input for meta model using base models' rescaled predictions
    inputs = pd.DataFrame()
    inputs['SVR'] = svr_pred
    inputs['NN'] = nn_pred
    inputs['MLP'] = mlp_pred
    inputs['XGB'] = xgb_pred
    # inputs['LSTM'] = inputs['LSTM'].shift(len(NN_predictions)-len(LSTM_predictions))
    # inputs = inputs[-len(LSTM_predictions):]

    
    # scale the rescaled predictions
    scaled_inputs = pre.scale(inputs)
    
    
    # predict
    pred = pre.predict(meta_model, scaled_inputs, df)
    
    # rescale the meta model's predictions and convert it into a list (with 1 value)
    pred = pre.rescale(pred)
    
    # return a value for the predicted volatility of the next timestep in a list
    return pred


# Run main() every hour

print(main(data))


# # Rank all the stocks based on predicted volatility in descending order
# if __name__ == "__main__":
  
    