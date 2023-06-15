from abc import ABC, ABCMeta, abstractmethod

import pandas as pd

class BaseMlModel(ABC):

    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def model_full_name(self) -> str:
        pass

    @abstractmethod
    def fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
        pass
    
    @abstractmethod
    def load(self, label:str, path:str) -> None:
        pass

    @abstractmethod
    def save(self, path:str) -> str:
        pass

    @abstractmethod
    def predict(self, data:pd.DataFrame) -> pd.Series:
        """Predicts the score (label variable) using passed data"""
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass