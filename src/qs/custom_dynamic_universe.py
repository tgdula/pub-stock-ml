import copy as cp
import logging
from typing import List

import pandas as pd
from qstrader.asset.universe.universe import Universe

import qs.defaults as default
from data.contracts import Columns


class CustomDynamicUniverse(Universe):
    """
    Generates a dynamic asset universe on a given date
    .. using custom, ML-powered scoring system
    """
    def __init__(self, score_data:pd.DataFrame, **kwargs):
        self.price_threshold_min = kwargs['price_threshold_min'] if 'price_threshold_min' in kwargs else default.MIN_ASSET_PRICE 
        self.price_threshold_max = kwargs['price_threshold_max'] if 'price_threshold_max' in kwargs else default.MAX_ASSET_PRICE 
        self.return_threshold = kwargs['return_threshold'] if 'return_threshold' in kwargs else default.MIN_RETURN_THRESHOLD

        self.universe = self._calculate_daily_universe(score_data)
        self.last_index_find_method = 'ffill'
        logging.info(f'CustomDynamicUniverse initialized: return_threshold={self.return_threshold}')


    def get_assets(self, dt:pd.Timestamp) -> List[str]:
        """
        Returns the selected universe from the score data
        NOTE: ensure to return none if there's nothing to return (e.g. bad market)
        NOTE: use `pd.index.get_loc(..,method=ffill)` to select this or previous data
        """
        found_index = self.universe.index.get_loc(dt, method=self.last_index_find_method) # type: ignore
        selected_universe = self.universe.iloc[found_index]
        logging.info(f"CustomDynamicUniverse selected ({dt}) assets: {str(selected_universe)}")
        return selected_universe.split(',') if selected_universe else []
    
    def _calculate_daily_universe(self, score_data:pd.DataFrame) -> pd.DataFrame:
        """
        Converts the provided score data to create a daily universe
        - penny_stocks_skip_lower_than - skip lower than
        """       
        df_universe = cp.deepcopy(score_data)
        # NOTE: select dynamic stock universe based on defined criteria
        df_universe = df_universe.where(
            # basic price criteria
            df_universe[Columns.PRICE].gt(self.price_threshold_min) & # NOTE: avoid penny stocks
            df_universe[Columns.PRICE].lt(self.price_threshold_max) & # NOTE: avoid too pricey stocks
            df_universe[Columns.SCORE].gt(self.return_threshold)      # NOTE: this to be tested
        )
        df_universe.reset_index(level=Columns.STOCK, inplace=True)
        df_universe = df_universe.dropna()                                # NOTE: crucial, as otherwise there're duplicates (!)
        df_universe = df_universe.groupby(Columns.DATE)[Columns.STOCK].agg( ','.join) # NOTE: convert to comma-separated list
        return df_universe
    