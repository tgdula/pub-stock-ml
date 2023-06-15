import datetime as dt
import pytest

from qstrader.risk_model.risk_model import RiskModel
from qs.risk_model import CustomRiskModel

MAX_POSITION_PERCENTAGE = 0.2
@pytest.mark.parametrize(
    'weights,weights_expected',
    [
        ({},{}),
        ({'A': 0.0, 'B': 0.0, 'C': 0.0},{'A': 0.0, 'B': 0.0, 'C': 0.0}),
        ({'A': 0.75, 'B': 0.25, 'C': 0.0},{'A': MAX_POSITION_PERCENTAGE, 'B': MAX_POSITION_PERCENTAGE, 'C': 0.0}),
    ]
)
def test_risk_model(weights,weights_expected):
    risk_model = CustomRiskModel(max_position_percentage=MAX_POSITION_PERCENTAGE)
    weights_converted = risk_model(dt.datetime.now(), weights)

    assert weights_converted == weights_expected