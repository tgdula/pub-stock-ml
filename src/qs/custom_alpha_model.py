import copy as cp
import itertools
import logging

import pandas as pd
from qstrader.alpha_model.alpha_model import AlphaModel
from qstrader.asset.universe.universe import Universe

import qs.defaults as default
from data.contracts import Columns
from qs.data_source import DataSourceWrapper


class CustomAlphaModel(AlphaModel):
    """
    Generates asset weights on a given date
    .. using custom, ML-powered dynamic universe
    .. so these weights can be used for portfolio re-allocation
    """
    def __init__(
        self, 
        score_data:pd.DataFrame, 
        universe:Universe, 
        data_source:DataSourceWrapper, 
        top_n:int=7, 
        max_pos:float=0.2, 
        **kwargs
    ):
        self.universes = [universe]
        self.data_source = data_source
        self.top_n = top_n
        self.max_pos = max_pos
        self.idx = pd.IndexSlice
        self.return_threshold = kwargs['return_threshold'] if 'return_threshold' in kwargs else default.MIN_RETURN_THRESHOLD
        self.score_data = self._init_data(score_data)
        logging.info(f'CustomAlphaModel initialized: n_pos:{self.top_n}| position_max:{self.max_pos}| return_threshold: {self.return_threshold}')
        
    def add_universe(
        self, universe:Universe
    ):
        self.universes.append(universe)


    def _init_data(self, data:pd.DataFrame) -> pd.Series:
        selected_data = cp.deepcopy(data.loc[self.idx[:,:],:])
        score_data = selected_data[Columns.SCORE]
        return score_data

    def _get_assets(
        self,dt
    ):
        # NOTE: consider multiple universes (could be: portfolio stocks / twitter recommendation / sentiment, favorite stocks, etc)
        assets = [universe.get_assets(dt) for universe in self.universes]
        assets = list(itertools.chain.from_iterable(assets))
        return assets
    
    
    def _calculate_weights_with_equal_max_position_limit(
        self, assets
    ):
        weights = {asset: min(self.max_pos,1/len(assets)) for asset in assets} if assets else {}
        return weights


    def __call__(
        self, dt:pd.Timestamp
    ):
        """
        Returns weights for assets
        """
        # NOTE: that's likely a `qstrader` specific work-around for unable to select time-part, together with:
        #       NotImplementedError: only the default get_loc method is currently supported for MultiIndex
        date_for = dt.replace(hour=0, minute=0, second=0)
        assets = self._get_assets(date_for)
        if not assets:
            return {}
        
        top_assets = []
        try:
            candidate_assets = list(
            self.score_data
                # NOTE: add filtering using some threshold value to make sure it's still positive
                #       this is expecially important when extended the selection with some extra universe
                #       portfolio stocks / twitter recommendation / sentiment, etc
                [self.score_data > self.return_threshold] # NOTE: above threshold
                .loc[self.idx[date_for:date_for,assets]]  # NOTE: selected assets only
                .sort_values(ascending=False)             # NOTE: for n-best selection
                .reset_index(level=1)                     # NOTE: take only stock-names
                [Columns.STOCK].values
            )
            logging.info(f"({date_for}) CustomAlphaModel candidate assets ({len(candidate_assets)}): {str(candidate_assets)} [from:{str(assets)}]")
            top_assets = candidate_assets[:self.top_n]

        except KeyError:
            print('Error occurred')
            logging.error(f'({date_for}) CustomAlphaModel error occurred')
            top_assets = []
        
        # NOTE: don't allow "all-in", but rather invest no more than `max_pos` amount
        #       could also be handled with "RiskModel" (qstrader)
        weights = self._calculate_weights_with_equal_max_position_limit(top_assets)
        
        logging.info(f"({date_for}) CustomAlphaModel selected assets: {str(weights)} from universe: {str(assets)}")
        return weights