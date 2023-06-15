import functools
import logging

import copy as cp
import datetime as dt
import numpy as np
import pandas as pd

from typing import List

class DataSourceWrapper:
    """
        Wrapper class for the dataset to provide historical closing prices, 
        similar to the `CSVDailyBarDataSource`, with `get_assets_historical_closes` method 

        NOTE: although ask/bid prices are separated (as 'in real'), 
              they in fact return the same values (unlike the 'real' market)
    """
    def __init__(self, prices:pd.Series, returns:pd.Series):
        self.date_format = '%Y-%m-%d'
        self.last_index_find_method = 'ffill'

        self.prices = self._transpose(prices)
        self.returns = self._transpose(returns)

    def _transpose(self, series:pd.Series) -> pd.DataFrame:
        data_copy = cp.deepcopy(series)
        data_copy = data_copy.unstack(level=1)
        data_copy.index = pd.to_datetime(data_copy.index, utc=True)
        return data_copy

    def get_asset_historical_closes(self, start_dt:dt.datetime, end_dt:dt.datetime, assets:List[str]) -> pd.DataFrame:
        logging.info(f'DataSourceWrapper: fetching historical prices {start_dt}-{end_dt}')
        to_index = self.prices.index.get_loc(end_dt, method=self.last_index_find_method) # type: ignore
        if start_dt is None:
            return self.prices.iloc[:to_index][assets]

        from_index = self.prices.index.get_loc(start_dt, method=self.last_index_find_method) # type: ignore
        return self.prices.iloc[from_index:to_index][assets]
    
    def get_asset_historical_returns(self, start_dt:dt.datetime, end_dt:dt.datetime, assets:List[str]) -> pd.DataFrame:
        to_index = self.returns.index.get_loc(end_dt, method=self.last_index_find_method) # type: ignore
        if start_dt is None:
            return self.returns.iloc[:to_index][assets]

        from_index = self.returns.index.get_loc(start_dt, method=self.last_index_find_method)        # type: ignore
        return self.returns.iloc[from_index:to_index][assets]

    def get_ask(self, date_time:dt.datetime, asset_symbol:str) -> float: # HINT: called from `BacktestDataHandler` (not: `get_asset_latest_ask_price`)
        return self._get_price(date_time, asset_symbol)

    def get_bid(self, date_time:dt.datetime, asset_symbol:str) -> float: # HINT: called from `BacktestDataHandler` (not: `get_asset_latest_bid_price`)
        return self._get_price(date_time, asset_symbol)

    #@functools.lru_cache(maxsize=1024 * 1024)
    def get_prices_on_date(self, date_time:dt.datetime,assets:List[str]):
        found_index = self.prices.index.get_loc(date_time, method=self.last_index_find_method) # type: ignore
        prices = self.prices.iloc[found_index][assets]
        if prices.empty:
            print(f"WARNING: prices not found at {date_time}. Taking previous-day")
        return prices if not prices.empty else self.get_prices_on_date(date_time - dt.timedelta(days=1), assets)
    
    @functools.lru_cache(maxsize=1024 * 1024)
    def _get_price(self, date_time:dt.datetime, asset_symbol:str) -> float:
        found_index = self.prices.index.get_loc(date_time, method=self.last_index_find_method) # type: ignore
        price = self.prices.iloc[found_index][asset_symbol]
        if np.isnan(price):
            print(f"WARNING: {asset_symbol} price not found at {date_time}. Taking previous-day")
        return price if not np.isnan(price) else self._get_price(date_time - dt.timedelta(days=1), asset_symbol)