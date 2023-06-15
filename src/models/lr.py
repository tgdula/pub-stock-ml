import logging
import os
import pickle
from typing import List, Optional

import copy as cp
import pandas as pd
import scipy.stats as scs
import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler

from models.base import BaseMlModel

class LinearRegressionModel(BaseMlModel):

    VERSION_NUMBER = '0.0.1'
    
    @property
    def model_full_name(self):
        return f'{self.model_name}_ver{self.version_number}'

    def __init__(self, model_name:str, label:str, categorical_features:List[str]=[], **kwargs):
        self.model:skl.LinearRegression = None # type: ignore
        self.model_name = model_name
        self.label = label
        self.categorical_features = categorical_features
        self.version_number = kwargs.get('version_number', self.VERSION_NUMBER)
            
    def fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
        if not self.model: self.model = skl.LinearRegression()

        train_data = cp.copy(train_data).dropna()
        X_train = train_data.drop(columns=[self.label])
        y_train = train_data.loc[:, self.label]
        self.model.fit(X=X_train, y=y_train)

        test_data = cp.copy(test_data).dropna()
        X_test = test_data.drop(columns=[self.label])
        y_test = test_data.loc[:, self.label]
        y_pred = self.model.predict(X_test)
        y_test.to_frame(f'{self.label}_predicted').assign(y_pred=y_pred)
        self.model_score = scs.spearmanr(y_test, y_pred)[0]
        logging.info(f'training {self} score - {str(self.model_score)}')
        
    
    def load(self, label:str, path:str) -> None:
        self.model = pickle.load(open(path, 'rb'))
        self.label = label
        
    def save(self, path:str) -> str:
        pickle.dump(self.model, open(path, 'wb'))
        return path

    def predict(self, data:pd.DataFrame) -> pd.Series:
        """Predicts the score (label variable) using passed data"""
        to_predict = data.drop(columns=[self.label])
        predicted = self.model.predict(to_predict)
        logging.info(f'predicted LR {self}')
        return predicted # type: ignore

    def __str__(self) -> str:
        return f'{self.__class__.__name__}:{self.model_name}-{self.version_number}| label:{self.label}'