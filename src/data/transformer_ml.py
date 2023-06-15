import logging
from typing import List,Tuple

import copy as cp
import pandas as pd
import quantstats as qs

from data.contracts import Columns
import data.transformer as tr 
import data.transformer_combined as trc

import models.base as ml


class MlTransformer(tr.BaseOhlcTransformer):
    def __init__(self, ml_model:ml.BaseMlModel, drop_raw_columns:bool=True):
        super().__init__()
        self.ml_model = ml_model
        self.drop_raw_columns=drop_raw_columns
        self.close = None

    def transform(self, data:pd.DataFrame) -> pd.DataFrame:
        has_raw_columns = Columns.CLOSE in data.columns
        if has_raw_columns:
            self.close = data[Columns.CLOSE]
        if has_raw_columns and self.drop_raw_columns:
            data.drop(columns=self.raw_column_names, inplace=True)
        
        predicted = self.ml_model.predict(data)
        data[f'{self.ml_model.label}_predicted'] = predicted # type: ignore
        
        if has_raw_columns and self.drop_raw_columns:
            data[Columns.CLOSE] = self.close
        return data

    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:
        return self.transform(data)