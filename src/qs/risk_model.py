import datetime as dt
import numpy as np

from qstrader.risk_model.risk_model import RiskModel

import qs.defaults as default

class CustomRiskModel(RiskModel):
    """Customized risk model: makes sure no position is grater than certain threshold"""
    def __init__(self, **kwargs):
        self.max_position_percentage = kwargs['max_position_percentage'] if 'max_position_percentage' in kwargs else default.MAX_POSITION_PERCENTAGE  

    def __call__(self, dt:dt.datetime, weights:dict) -> dict:
        if self._has_zero_weights(weights): # no position means no risk
            return weights
        resulting_weights = {
            asset : min(weight, self.max_position_percentage) for (asset,weight) in weights.items()}
        return resulting_weights
    
    def _has_zero_weights(self, weights:dict):
        weight_sum = sum(weight for weight in weights.values())
        return np.isclose(weight_sum, 0.0)