from os import listdir
from os.path import isfile, join, splitext
from typing import Optional

import numpy as np
import pandas as pd

from data.contracts import Columns

class DataSource:
    """
        Represents a source of all stock data
    """
    DATA_STORE = "/data/stooq/assets.h5"
    PRICE_DATA = 'pl/prices'
    TICKER_SYMBOLS_EXCLUDED = [sym for sym in [
    'FCHF','FCHFWS','FUSD','FUSDWS','FEUR','FEURWS','FW20WS',
    'INVESTORMS', # ValueError: invalid combination of [values_axes] on appending data
    'MWIG40DVP','MWIG40DVP','MWIG40TR','NCINDEX','RESPECT','SWIG80DVP','SWIG80TR']]
    INDEX_SYMBOL = 'WIG'
    
    def __init__(self, from_date:Optional[str]=None, to_date:Optional[str]=None, **kwargs):
        """
        NOTE: there are two types of params
        (1) domain-based: from/to dates, stock list, index, column names
        (2) physical storage, etc
        what about params: data source (path), data store (name), symbols excluded, index?"""
        self.data_store_path = kwargs.get('data_store_path', self.DATA_STORE)
        self.price_data_field = kwargs.get('price_data_field', self.PRICE_DATA)
        self.ticker_symbols_excluded = kwargs.get('ticker_symbols_excluded', self.TICKER_SYMBOLS_EXCLUDED)
        self.index_symbol = kwargs.get('index_symbol', self.INDEX_SYMBOL)

        with pd.HDFStore(self.data_store_path) as store:
            all_data = (store[self.price_data_field].dropna())
            all_data.sort_index(ascending=True, inplace=True)
            all_data = all_data[~all_data.index.get_level_values(Columns.STOCK).isin([self.ticker_symbols_excluded])]
            all_data = all_data[~all_data.index.get_level_values(Columns.STOCK).str.startswith(self.index_symbol)]
            all_data = self._filter_data(all_data, from_date=from_date, to_date=to_date) # type: ignore
            self.data = all_data
        
        # get_all_stock_names()
        self.from_date = from_date
        self.to_date = to_date
        self.symbols = self.data.index.get_level_values(Columns.STOCK).unique().sort_values()
        self.dates = self.data.index.get_level_values(Columns.DATE).unique().sort_values()

    @staticmethod
    def _filter_data(data:pd.DataFrame, from_date:Optional[str]=None, to_date:Optional[str]=None) -> pd.DataFrame:
        data = data[~data.index.duplicated(keep='first')]
        idx = pd.IndexSlice

        if from_date and to_date:
            data = data.loc[idx[from_date:to_date, :], :]
        elif from_date:
            data = data.loc[idx[from_date:, :], :]
        elif to_date:
            data = data.loc[idx[:to_date, :], :]
        return data

    def get_data(self, from_date:Optional[str]=None, to_date:Optional[str]=None) -> pd.DataFrame: 
        """Retrieves copy of data from given time period
        NOTE: data is multiindexed [Columns.DATE,Columns.STOCK]
        | idx=date | idx=stock | open | high | low | close | volume |
        in order to flatten the index, use: 
            data.reset_index(level=[0,1])   -> drop both indices (could also: inplace=True)
            data.reset_index(level=[1])     -> drop only 'stock' index
        to sort 
            data.sort_index()               -> (could also: inplace=True)
        """        
        data = self._filter_data(self.data, from_date, to_date).copy()
        return data
    
    def update(self, data:pd.DataFrame, date:str):
        idx = pd.IndexSlice
        # when the passed dataframe has data, and self.dataframe doesn't: concatenate
        if self.data.loc[idx[date:date, :], :].empty and not data.loc[idx[date:date, :], :].empty:
            self.data = pd.concat([self.data.reset_index(),data.reset_index()], ignore_index=True)
            self.data.set_index([Columns.DATE,Columns.STOCK],inplace=True)
            self.data.sort_index(ascending=True, inplace=True)

    def get_asset(self, asset:str, from_date:Optional[str]=None, to_date:Optional[str]=None) -> pd.DataFrame:
        idx = pd.IndexSlice
        data = self.get_data(from_date, to_date).loc[idx[:, asset], :]
        return data
