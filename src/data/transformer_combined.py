import copy as cp
import logging
import pandas as pd
import quantstats as qs

from typing import List,Tuple

from data.contracts import Columns
import data.transformer as tr 

###
### Combined transformers: Technicals, signals, Trends + generic: PipelineTransformer, ResultTransformer
### 

class TechnicalIndicators():
    def __init__(self, transformers:List[tr.BaseOhlcTransformer]=[]):
        self.transformers = transformers if transformers else [tr.BollingerBands(), tr.Natr(), tr.Ppo(), tr.Mfi()]
    
    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:
        for transform in self.transformers:
            data = transform(data)
        logging.debug(f'calculated technical indicators')
        return data


class Trends():
    def __init__(self, transformers:List[tr.BaseOhlcTransformer]=[]):
        self.transformers = transformers if transformers else [tr.LinearRegression()]
    
    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:
        for transform in self.transformers:
            data = transform(data)
        
        logging.debug(f'calculated trend characteristics')
        return data
    
class PipelineTransformer(tr.BaseOhlcTransformer):
    """
    A composite transformer, that allows creating pipelines of internal ones
    """        
    def __init__(self, transformers:List[tr.BaseOhlcTransformer]=[]): 
        super().__init__()
        # HINT: following is required to work-around issue when instantiating from a configuration
        self.transformers = transformers

    def transform(self, data:pd.DataFrame, drop_raw_columns:bool=True) -> Tuple[pd.DataFrame, str]:
        """
        Transforms OHLC price data into
        - set of features for machine learning algorithm
        - target label for supervised learning
         """
        prices = cp.deepcopy(data)
        
        for transform in self.transformers:
            prices = transform(prices)
        logging.debug(f'calculated data')

        # calculate target label (e.g. Return_x)
        target, target_label = self._calculate_return_target(prices)
        prices[target_label] = target
        logging.debug(f'calculated target')

        # drop initial raw columns (not needed for prediction)
        if drop_raw_columns:
            prices.drop(columns=self.raw_column_names, inplace=True)
        
        return prices, target_label
    
    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:
        for transform in self.transformers:
            data = transform(data)
        return data
   

    def _calculate_return_target(self, data:pd.DataFrame, return_lag:int = 5, winsorization_threshold:float = 0.001) -> Tuple[pd.Series, str]:
        target = (data.groupby(level=Columns.STOCK).price
                                .pct_change(return_lag)
                                .pipe(lambda x: x.clip(lower=x.quantile(winsorization_threshold),
                                                       upper=x.quantile(1 - winsorization_threshold)))
                                .add(1)
                                .pow(1 / return_lag)
                                .sub(1)
                                )
        target_label = f'return_{return_lag}d'
        logging.debug(f'calculated target: {target_label}')

        return target, target_label
    
    
class ResultTransformer(PipelineTransformer):
    def __init__(
            self,
            return_lags:List[int]=[1,5], 
            number_of_bins:int=7, 
            outlier_winsorize_quantile:float=0.0001,
            transformers:List[tr.BaseOhlcTransformer]=[]
    ):
        super().__init__()
        self.return_lags = return_lags
        self.number_of_bins = number_of_bins
        self.outlier_winsorize_quantile = outlier_winsorize_quantile
        self.transformers = transformers if transformers else [
            tr.SortinoTransformer(
                number_of_bins=number_of_bins,
                outlier_winsorize_quantile=outlier_winsorize_quantile, 
                return_lags=return_lags), 
            tr.StopLossCalculator()
        ]

        qs.extend_pandas()

    def transform(self,data:pd.DataFrame) -> pd.DataFrame:
        data = cp.deepcopy(data).dropna()
        for transform in self.transformers:
            data = transform(data)
        return data
    
    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:
        return self.transform(data)