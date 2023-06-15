import logging
import os
from typing import List, Optional  # NOTE: until migrated to Python 3.9

import lightgbm as lgb
import pandas as pd
import scipy.stats as scs

from models.base import BaseMlModel

# import sklearn.metrics as skm

    
class LgbmModel(BaseMlModel):
    DEFAULT_MODEL_NAME = 'lgbm_model'
    FEATURE_FRACTION = 0.95
    LEARNING_RATE = 0.01
    MIN_DATA_IN_LEAF = 250
    NUM_BOOST_ROUND = 25 # 100
    MAX_DEPTH = 8       # recommended: a value from range (3..12)
    NUM_LEAVES = 256    # recommended: 2^(max_depth)

    VERSION_NUMBER = '0.0.1'

    @property
    def model_full_name(self):
        return f'{self.model_name}_ver{self.version_number}'

    def __init__(self, model_name:str, label:str, categorical_features:List[str]=[], **kwargs):
        """
        LightGBM Model

        Initialize with a dictionary with parameters (see LightGBM documentation), 
        Actual parameters should be the result of optimization - here are ones : {
            'num_boost_round': 25,
            'num_leaves': 4
            'learning_rate': 0.01,
            'feature_fraction': 0.95,
            'min_data_in_leaf': 250
        }
        """
        self.num_boost_round = kwargs.get('num_boost_round', self.NUM_BOOST_ROUND)
        self.model_params = {
            'learning_rate': kwargs.get('learning_rate', self.LEARNING_RATE),
            'num_leaves': kwargs.get('num_leaves', self.NUM_LEAVES),
            'feature_fraction': kwargs.get('feature_fraction', self.FEATURE_FRACTION),
            'min_data_in_leaf': kwargs.get('min_data_in_leaf', self.MIN_DATA_IN_LEAF),
            'max_depth': kwargs.get('max_depth', self.MAX_DEPTH),
        }
        self.categorical_features = categorical_features
        self.label = label
        self.model:Optional[lgb.Booster] = None # lgb.Booster() # will be overwritten in train or load
        self.model_score = None # SpearmanrResult(correlation,pvalue)
        self.model_name = model_name
        self.version_number = kwargs.get('version_number', self.VERSION_NUMBER)
        logging.debug(f'{self.model_name} initialized with parameters {os.linesep}{self.model_params}')


    def fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
        
        logging.info(f'training {self.model_full_name} - starting..')
        lgb_data = lgb.Dataset(
                    data=train_data.drop(self.label, axis=1), 
                    label=train_data[self.label], 
                    categorical_feature=self.categorical_features,
                    free_raw_data=False)
        
        self.model = lgb.train(params=self.model_params,
                              train_set=lgb_data,
                              num_boost_round=self.num_boost_round,
                              verbose_eval=False)
        
        logging.info(f'training {self.model_full_name} - finished')
        
        # TODO: evaluate model: https://www.freecodecamp.org/news/evaluation-metrics-for-regression-problems-machine-learning/
        X_test = test_data.loc[:, self.model.feature_name()]
        y_test = test_data.loc[:, self.label]
        y_pred = self.model.predict(X_test)
        y_test.to_frame(f'{self.label}_predicted').assign(y_pred=y_pred)
        self.model_score = scs.spearmanr(y_test, y_pred)[0]
        logging.info(f'training {self.model_full_name} score - {str(self.model_score)}')
        # TODO: create metadata based on training (e.g. score (loss function))

        self.model_metadata = None 

    def load(self, model_full_path:str, label:str, model_name:Optional[str]=None):
        # TODO :try reading model name from metadata file or provide as param (consider classmethod)
        """
        @classmethod
        def load_from_file(cls, model_full_path:str, model_name:str=None):
            model = cls(...) >> but how to get the label name - metadata?        
        """
        self.model = lgb.Booster(model_file=model_full_path)
        self.label = label
        self.model_name = model_name

    def save(self, models_folder_full_path:str):
        assert self.model is not None
        
        model_fullpath_name = os.path.join(models_folder_full_path,self.model_full_name)
        model_file, metadata_file = f'{model_fullpath_name}.txt', f'{model_fullpath_name}_metadata.txt'
        self.model.save_model(model_file, num_iteration=self.model.best_iteration) # type: ignore
        with open(metadata_file, 'w') as metadata_file:
            metadata_file.write((
                f'{self.model_full_name}{os.linesep}{os.linesep}'
                f'score: {self.model_score}{os.linesep}'
                f'metadata:{os.linesep}{str(self.model_params)}{os.linesep}{os.linesep}'
                f'feature importance:{os.linesep}{str(self.get_feature_importance())}'))
        return model_file #, metadata_file

    def get_feature_importance(self):
        assert self.model is not None
        feature_importance = self.model.feature_importance(importance_type='gain')
        return (pd.Series(feature_importance / feature_importance.sum(),
            index=self.model.feature_name()))
    
    def predict(self, data:pd.DataFrame) -> pd.Series:
        predict_from = data if not self.label in data.columns else data.drop(columns=[self.label])
        return self.model.predict(predict_from) # type: ignore
    
    def __str__(self) -> str:
        return f'{self.__class__.__name__}:{self.model_name}-{self.version_number}| label:{self.label}'
