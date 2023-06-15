from abc import ABC, abstractmethod
from typing import List,Tuple

import logging
import os

import numpy as np
import pandas as pd
import quantstats as qs
import talib as ta

from data.contracts import Columns


class BaseOhlcTransformer(ABC):
    """
    OHLC(V) data can't be used directly (raw) in training (noise etc)
    It needs to be processed. See the other versions of the transformer class
    """
    def __init__(self):
        self.raw_column_names = [
            Columns.OPEN,
            Columns.HIGH,
            Columns.LOW,
            Columns.CLOSE,
            Columns.VOLUME
        ]
    
    @abstractmethod
    def transform(self, data:pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """
        Transforms OHLC price data into
        - set of features for machine learning algorithm
        - target label for supervised learning
        NOTE: data is multi-indexed - handles that
         """
        pass

    @abstractmethod
    def __call__(self, dt):
        raise NotImplementedError(
            "Should implement __call__()"
        )


class Preprocessor():
    def _calculate_adjusted_price(self, data:pd.DataFrame) -> pd.Series:
        """Calculates price as weighted average: HLCC/4 (takes high & low, but close price is most important)"""
        result = (data[Columns.HIGH] + data[Columns.LOW] + data[Columns.CLOSE] * 2) / 4
        return result    
    
    def _calculate_z_score(self, serie:pd.Series) -> pd.Series:
        """ z = X - μ / σ """
        result = (serie - serie.rolling(window=200, min_periods=20).mean()) / serie.rolling(window=200, min_periods=20).std()
        return result
    
    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:
        data['price'] = data.groupby(level=Columns.STOCK, group_keys=False).apply(self._calculate_adjusted_price)
        data['volume_z_score'] = data.groupby(level=Columns.STOCK).volume.apply(self._calculate_z_score)
        return data
    

class DataCleaner():
    """Removes data with too-few observations and illiquid ones"""
    def __init__(self, price_column:str=Columns.CLOSE, observations_threshold:float=0.8):
        self.price_column = price_column
        self.observations_threshold = observations_threshold

    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:
        
        idx = pd.IndexSlice
        
        # drop stocks with less of threshold (e.g. 80%) of observations - otherwise errors in signals calculation
        nobs = data.groupby(level=Columns.STOCK).size()
        nobs_all = nobs.max()
        nobs_threshold = int(nobs_all * self.observations_threshold)
        min_obs = nobs_threshold
        keep = nobs[nobs > min_obs].index
        data = data.loc[idx[:, keep], :]
        logging.debug(f'removed data with insufficient data')

        # calculate  dollar_vol_rank, drop stocks with too few liquidity
        # HINT: use STD instead of specific, hardcoded threshold (better filters-out)
        # TODO: rename to "liquidity" and "liquidity_rank"
        data['dollar_vol'] = data.loc[:, self.price_column].mul(data.loc[:, Columns.VOLUME], axis=0) # type: ignore
        data['dollar_vol_rank'] = (data
                             .groupby(Columns.DATE)
                             .dollar_vol
                             .rank(ascending=False))
        all_symbols = set(data.index.get_level_values(Columns.STOCK))
        dollar_vol_ranks = data.groupby(level=Columns.STOCK).dollar_vol_rank.mean()
        rank_threshold = dollar_vol_ranks.mean() + (dollar_vol_ranks.std() / 4) # NOTE: mere std() isn't enough (real-world tested!)

        keep = dollar_vol_ranks[dollar_vol_ranks <= rank_threshold].index
        data = data.loc[idx[:, keep], :]
        data.drop(columns=['dollar_vol'], inplace=True)        
        remaining_symbols = set(data.index.get_level_values(Columns.STOCK))

        removed_synbols = sorted(all_symbols - remaining_symbols)
        number_of_all = len(all_symbols)        
        number_of_removed = len(removed_synbols)
        logging.debug(
            f'removed illiquid stocks data ({number_of_removed} of {number_of_all}){os.linesep}'
            f'{removed_synbols}')

        logging.info(f'removed data with insufficient liquidity: {rank_threshold} of {len(all_symbols)}: {removed_synbols}')

        # logging.debug(
        #     f'removed illiquid stocks data ({len(all_symbols)-len(remaining_symbols)} of {len(all_symbols)}){os.linesep}'
        #     f'{all_symbols - remaining_symbols}')

        return data

class DataImputer():
    """Provides missing data"""
    def __init__(self, method:str='ffill'):
        self.method = method
    
    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:
        return data.unstack(1).fillna(method=self.method).stack('stock')  # type: ignore

###
### single indicators
### 

class LinearRegression():
    def __init__(self, column:str='lin_reg'):
        self.column = column 

    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:
        def compute_linear_regression(price:pd.Series) -> pd.Series:
            slope = ta.LINEARREG_SLOPE(price, timeperiod=14) # type: ignore (Pylance issue with ta-lib)
            angle = np.rad2deg(np.arctan(slope))
            return angle
        data[self.column] = data.groupby(level=Columns.STOCK, group_keys=False).price.apply(compute_linear_regression)
        return data    

    
class BollingerBands():
    def __init__(self, columns:List[str]=['bb_high','bb_low']):
        self.columns = columns
        self.column = ','.join(self.columns)
    
    def _calculate_bb(self, price):
            high, mid, low = ta.BBANDS(price, timeperiod=20) # type: ignore
            return pd.DataFrame({self.columns[0]: high, self.columns[1]: low}, index=price.index)

    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:        
        data = data.join(data.groupby(level=Columns.STOCK).price.apply(self._calculate_bb))
        data[self.columns[0]] = data.bb_high.sub(data.price).div(data.bb_high).apply(np.log1p)
        data[self.columns[1]] = data.price.sub(data.bb_low).div(data.price).apply(np.log1p)
        return data
    

class Natr():
    def __init__(self, column:str='natr'):
        self.column = column 

    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:
        def calculate_natr(data:pd.DataFrame) -> pd.Series:
            df = ta.NATR(data.high, data.low, data.close, timeperiod=14) # type: ignore (Pylance issue with ta-lib)
            return df.sub(df.mean()).div(df.std())
        data[self.column] = (data.groupby(Columns.STOCK, group_keys=False).apply(calculate_natr))
        return data
    

class Ppo():
    def __init__(self, column:str='ppo'):
        self.column = column 

    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:
        data[self.column] = data.groupby(Columns.STOCK, group_keys=False).price \
            .apply(lambda x: ta.PPO(x, fastperiod=12, slowperiod=26, matype=0)) # type: ignore (Pylance issue with ta-lib)
        return data
    

class Mfi():
    def __init__(self, column:str='mfi'):
        self.column = column 

    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:
        data[self.column] = data.groupby(Columns.STOCK, group_keys=False) \
            .apply(lambda x: ta.MFI(x.high, x.low, x.close, x.volume, timeperiod=14)) # type: ignore (Pylance issue with ta-lib)
        return data


class Adx():
    def __init__(self, column:str='adx', timeperiod:int=14):
        self.column = column
        self.timeperiod=timeperiod

    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:
        data[self.column] = data.groupby(Columns.STOCK, group_keys=False) \
            .apply(lambda x: ta.ADX(x.high, x.low, x.close, timeperiod=self.timeperiod) / ta.ADX(x.high, x.low, x.close, timeperiod=self.timeperiod).mean()) # type: ignore (Pylance issue with ta-lib)
        return data

class Adxr():
    def __init__(self, column:str='adxr', timeperiod:int=14):
        self.column = column
        self.timeperiod=timeperiod

    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:
        data[self.column] = data.groupby(Columns.STOCK, group_keys=False) \
            .apply(lambda x: ta.ADXR(x.high, x.low, x.close, timeperiod=self.timeperiod) / ta.ADXR(x.high, x.low, x.close, timeperiod=self.timeperiod).mean()) # type: ignore (Pylance issue with ta-lib)
        return data

class Atr():
    def __init__(self, column:str='atr', timeperiod:int=14):
        self.column = column
        self.timeperiod=timeperiod

    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:
        data[self.column] = data.groupby(Columns.STOCK, group_keys=False) \
            .apply(lambda x: ta.ATR(x.high, x.low, x.close, timeperiod=self.timeperiod) / ta.ATR(x.high, x.low, x.close, timeperiod=self.timeperiod).mean()) # type: ignore (Pylance issue with ta-lib)
        return data

class Rsi():
    def __init__(self, column:str='rsi', timeperiods:List[int]=[6,12]):
        self.column = column
        self.timeperiods=timeperiods

    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:
        for timeperiod in self.timeperiods:
            data[f'{self.column}{timeperiod}'] = data.groupby(Columns.STOCK, group_keys=False).price \
                .apply(lambda x: ta.RSI(x, timeperiod=timeperiod)) # type: ignore (Pylance issue with ta-lib)
        return data


class Ema():
    def __init__(self, column:str='ema', timeperiods:List[int]=[5,20], relative_timeperiod:int=50):
        self.column = column
        self.timeperiods=timeperiods
        self.relative_timeperiod = relative_timeperiod

    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:
        for timeperiod in self.timeperiods:
            data[f'{self.column}{timeperiod}'] = data.groupby(Columns.STOCK, group_keys=False).price \
                .apply(lambda x: ta.EMA(x, timeperiod=timeperiod) / ta.EMA(x, timeperiod=self.relative_timeperiod)) # type: ignore (Pylance issue with ta-lib)
        return data


class WillR():
    def __init__(self, column:str='willr', timeperiods:List[int]=[6,12]):
        self.column = column
        self.timeperiods=timeperiods

    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:
        for timeperiod in self.timeperiods:
            data[f'{self.column}{timeperiod}'] = data.groupby(Columns.STOCK, group_keys=False) \
                .apply(lambda x: ta.WILLR(x.high, x.low, x.close, timeperiod=timeperiod)) # type: ignore (Pylance issue with ta-lib)
        return data
    
    
class RateOfChange():
    def __init__(self, column:str='roc_', related_indicators:List[str]=[], timeperiod:int=2):
        self.column_prefix = column
        self.related_indicators = related_indicators
        self.timeperiod=timeperiod

    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:
        for col in self.related_indicators:
            data[f'{self.column_prefix}{col}'] = data.groupby(Columns.STOCK, group_keys=False)[col] \
                .apply(lambda x: ta.ROC(x, timeperiod=self.timeperiod)) # type: ignore (Pylance issue with ta-lib)
        return data
    

class SortinoTransformer():
    def __init__(self, number_of_bins:int=7, outlier_winsorize_quantile:float=0.0001, return_lags:List[int]=[1,5]):
        self.number_of_bins=number_of_bins
        self.return_lags = return_lags
        self.outlier_winsorize_quantile = outlier_winsorize_quantile

    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:
        for lag in self.return_lags:
            data[f'return_{lag}d'] = (data.groupby(level=Columns.STOCK).price
                                        .pct_change(lag)
                                        .pipe(lambda x: x.clip(lower=x.quantile(self.outlier_winsorize_quantile),
                                                            upper=x.quantile(1 - self.outlier_winsorize_quantile)))
                                        .add(1)
                                        .pow(1 / lag)
                                        .sub(1)
                                        )
            
            # HINT: that requires quantstats (called internally with `qs.extend_pandas()`)
            data[f'sortino_return_{lag}d'] = data.groupby(level=Columns.STOCK)[f'return_{lag}d'].apply(lambda x: x.rolling_sortino())
            data[f'sortino_return_{lag}d_bin']= pd.qcut(
                data[f'sortino_return_{lag}d'], 
                q = self.number_of_bins, 
                labels=[i for i in range(self.number_of_bins,0,-1)], 
                retbins=False) # labels=False makes sure that the column contains the index of the quartile, not the values.
        return data


class RankNormalizer(): # HINT: unreliable
    def __init__(self):
        pass

    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:
        def _normalize_vol_ranks(series:pd.Series) -> pd.Series:
            """Calculates price as weighted average: HLCC/4 (takes high & low, but close price is most important)"""
            result = 1-(series-series.min())/(series.max()-series.min())
            return result    
    
        # calculate volume-rank (bins vs normalize
        data[f'dollar_vol_rank_norm'] = data.groupby(level=Columns.STOCK).dollar_vol_rank.apply(_normalize_vol_ranks)

        return data
    

class StopLossCalculator():
    def __init__(self, stop_loss_period:int=2, stop_loss_percentages:List[int]=[10,15]):
        self.stop_loss_period = stop_loss_period
        self.stop_loss_percentages = stop_loss_percentages

    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:
        def _calculate_stop_price(series:pd.Series, stop_loss_period:int=2, stop_loss_threshold:float=0.15) -> pd.Series:
            """Calculates stop-loss-price"""
            price_fraction_threshold = 1-stop_loss_threshold
            result = series.rolling(stop_loss_period).max() * price_fraction_threshold
            return result    
    
        # calculate stop loss - price thresholds to be selected 
        for stop_loss_percentage in self.stop_loss_percentages:
            data[f'close_{stop_loss_percentage}'] = data.groupby(level=Columns.STOCK).price.apply(
                lambda x: _calculate_stop_price(x, self.stop_loss_period, stop_loss_percentage/100))
        return data