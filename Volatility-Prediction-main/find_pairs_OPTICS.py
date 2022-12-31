import matplotlib.pyplot as plt
import matplotlib.cm as cm
# import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing
# from statsmodels.tsa.stattools import coint, adfuller
from scipy import stats
# import requests
# from bs4 import BeautifulSoup
# import time
# import pymc3 as pm
# import theano as th
# import seaborn as sns
import missingno
# import pandas_datareader as pdr
# import yfinance as yf
# from itertools import combinations
# import yahoo_fin.stock_info as si
# import statsmodels.api as sm


trade_start="2019-1-1"

class find_pairs:
    
    # used to get the individual stocks in each pair
    def parse_pair(self, pair):
        s1 = pair[:pair.find('-')]
        s2 = pair[pair.find('-')+1:]
        return s1,s2
    
    
    def get_tradeable_pairs(self):
        """1.Apply PCA     2.Cluster the stocks with OPTICS    3.Get all possible pairs' symbols    4.Find the top 5 pairs with the highest Kendall's tau value"""
        
    
        """Read the csv file"""
        data = pd.read_csv(r'S&P500_stock_returns.csv', parse_dates=['Date'], index_col=['Date'])
        # split_date = pd.datetime(2021,1,1)
        # data = data.loc[trade_start : split_date]
        # Display missing data chart
        missingno.matrix(data)
        
        
        # Drop columns with more than 10% missing data
        missing_percentage = data.isnull().mean().sort_values(ascending=False)
        dropped_list = sorted(list(missing_percentage[missing_percentage > 0.1].index))
        data.drop(labels=dropped_list, axis=1, inplace=True)
        
        # Remove nan
        returns = data.iloc[1: , :]
        # print(returns)
        for i in returns.columns:
            returns[i] = returns[i].fillna(returns[i].mean())
        
        
        # Scale the returns data
        returns_scaled = preprocessing.StandardScaler().fit_transform(returns)
        # print(returns_scaled.shape)
        
        # Make columns the tickers and the values the scaled returns
        returns_scaled_df = pd.DataFrame(returns_scaled, columns=returns.columns)
        # print(returns_scaled_df)
        
        
        """Plot PCA explained variance graph""" # for finding the best number of components
        # pca = PCA().fit(returns)
        
        # plt.figure()
        # plt.plot(np.cumsum(pca.explained_variance_ratio_))
        # plt.xlabel('Number of Components')
        # plt.ylabel('Variance (%)')T
        # plt.title('Explained Variance')
        # plt.show()
        

        """PCA"""
        N_PRIN_COMPONENTS = 30
        pca = PCA(n_components=N_PRIN_COMPONENTS)
        pca.fit(returns_scaled)
        # print(pca.components_.T.shape)
        
        X = pca.components_.T
        
        
        """OPTICS clustering"""
        # clf = DBSCAN(eps=1.9, min_samples=3)
        clf = OPTICS(min_samples=5, xi=0.05).fit(X)
        print(clf)
        labels = clf.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print("\nClusters discovered: %d" % n_clusters_)
        clustered = clf.labels_
        
        # the initial dimensionality of the search was
    
        ticker_count = len(returns_scaled_df.columns)
        print("Total pairs possible in universe: %d " % (ticker_count*(ticker_count-1)/2))
        
        clustered_series = pd.Series(index=returns_scaled_df.columns, data=clustered.flatten())
        clustered_series_all = pd.Series(index=returns_scaled_df.columns, data=clustered.flatten())
        clustered_series = clustered_series[clustered_series != -1]
        CLUSTER_SIZE_LIMIT = 9999
        counts = clustered_series.value_counts()
        ticker_count_reduced = counts[(counts>1) & (counts<=CLUSTER_SIZE_LIMIT)]
        print("Clusters formed: %d" % len(ticker_count_reduced))
        print("Pairs to evaluate: %d" % (ticker_count_reduced*(ticker_count_reduced-1)).sum())
        
        
        """Display TSNE results of the Clustering"""
        X_tsne = TSNE(learning_rate=1000, perplexity=25, random_state=1337).fit_transform(X)
        # plt.figure(1, facecolor='white')
        plt.figure(figsize=(16, 8), facecolor='black')
        plt.clf()
        plt.axis('off')
        plt.scatter(
            X_tsne[(labels!=-1), 0],
            X_tsne[(labels!=-1), 1],
            s=100,
            alpha=0.85,
            c=labels[labels!=-1],
            cmap=cm.Paired
        )
        plt.scatter(
            X_tsne[(clustered_series_all==-1).values, 0],
            X_tsne[(clustered_series_all==-1).values, 1],
            s=100,
            alpha=0.05
        )
        plt.title('T-SNE of all Stocks with DBSCAN Clusters Noted');
        plt.show()
        
        
        """Plot out the stock prices of the chosen groups of stocks"""
        # get the number of stocks in each cluster
        counts = clustered_series.value_counts()
        # let's visualize some clusters
        cluster_vis_list = list(counts[(counts<20) & (counts>1)].index)[::-1]
        
        # plot clusters
        for clust in cluster_vis_list:
            tickers = list(clustered_series[clustered_series==clust].index)
            # Revert returns_scaled back to price
            price = 1*(1 + returns[tickers]).cumprod()
            price.plot(title='Stock Time Series for Cluster %d' % clust, linewidth=0.5)
            plt.show()
        
        
        """Find every possible pair and their kendall's tau value"""
        
        results = pd.DataFrame(columns=['tau'])
    
        # for each cluster, get the selected pairs
        for clust in cluster_vis_list:
            tickers = list(clustered_series[clustered_series==clust].index)
            cluster_df = pd.DataFrame(columns=tickers)
            for i in tickers:
                returns_list = returns[i]
                cluster_df[i] = returns_list
        
            # find each possible pair, and make it as the index of [results]
            for s1 in cluster_df.columns:
                for s2 in cluster_df.columns:
                    if (s1!=s2) and (f'{s2}-{s1}' not in results.index):
                        results.loc[f'{s1}-{s2}'] = stats.kendalltau(cluster_df[s1], cluster_df[s2])[0]
        
        
        # """Perform ARODS on each pair"""
        # for clust in cluster_vis_list:
        #     tickers = list(clustered_series[clustered_series==clust].index)
        #     res = list(combinations(tickers, 2))
        #     #Get the price data of each stock in the current iteration pair
        #     stock1 = res[0][0]
        #     stock2 = price_data[res[0][1]]
        #     model = sm.OLS(stock1, stock2)
        #     model = model.fit()
        #     print(model.params[0])
        #     spread = stock2 - model.params[0] * stock1
        #     adf = adfuller(spread, maxlag=1)
        #     if adf[0] < adf[4]['1%']:
                
        # # printing result 
        #     print("All possible pairs : " + str(res))
        
        
        # for clust in cluster_vis_list:
        #     tickers = list(clustered_series[clustered_series==clust].index)
        #     selected_stocks.append(tickers)
        
        

        """Get top 5 pairs with the highest kendall's tau value and each individual stock in the chosen pairs (for copula fitting)"""
        
        selected_pairs = []
        selected_stocks = []
        
        for pair in results.sort_values(by='tau', ascending=False).index: # get each stock pair(index of [results]) sorted by kendall's tau value
            s1,s2 = self.parse_pair(pair) # get individual stock symbols from the pairs
            if (s1 not in selected_stocks) or (s2 not in selected_stocks): 
                selected_stocks.append(s1)
                selected_stocks.append(s2)
                selected_pairs.append(pair)
            
            if len(selected_pairs) == 25: # get top 5 pairs with the highest 
                break
            
        selected_stocks = list(set(selected_stocks))
    
        tradeable_pairs = selected_pairs
        for pair in tradeable_pairs:
            stock1, stock2 = self.parse_pair(pair)
            price1 = 1*(1 + returns[stock1]).cumprod()
            price2 = 1*(1 + returns[stock2]).cumprod()
            plt.plot(price1, "orange")
            plt.plot(price2, "cyan")
            plt.show()

        return selected_pairs, selected_stocks
   