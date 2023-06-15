import logging
import pytz

import datetime as dt
import pandas as pd

from typing import Optional, Tuple

from qstrader.alpha_model.alpha_model import AlphaModel
from qstrader.risk_model.risk_model import RiskModel
from qstrader.asset.universe.universe import Universe
from qstrader.broker.fee_model.percent_fee_model import PercentFeeModel
from qstrader.broker.fee_model.zero_fee_model import ZeroFeeModel
from qstrader.data.backtest_data_handler import BacktestDataHandler
from qstrader.trading.backtest import BacktestTradingSession
import qstrader.statistics.performance as perf
from qstrader.statistics.tearsheet import TearsheetStatistics


import qs.defaults as default
import utils.date as dut 
from data.contracts import Columns

from qs.data_source import DataSourceWrapper
from qs.order_sizer import CustomizedOrderSizer

class Experiment():
    def __init__(
        self, 
        data:pd.DataFrame, 
        from_date:str, 
        to_date:str, 
        universe:Universe,
        data_source:DataSourceWrapper,
        alpha_model:AlphaModel,
        initial_cash:float=10000.00,
        risk_model:Optional[RiskModel]=None,
        **kwargs
        ):
        self.logger = logging.getLogger('Experiment')
        
        # NOTE: initialize dates (check whether include burn-in period)
        no_burn_in_date = kwargs.get('no_burn_in_date', False)
        self.start_dt = pd.Timestamp(from_date, tz=pytz.UTC)
        self.end_dt = pd.Timestamp(to_date, tz=pytz.UTC)
        if no_burn_in_date:
            self.burn_in_dt = self.start_dt
        else:
            burn_in = dut.to_date(dut.adjust_days(from_date, 9 * 30))
            self.burn_in_dt = pd.Timestamp(burn_in.strftime(dut.date_format), tz=pytz.UTC)

        self.rebalance=kwargs.get('rebalance', default.REBALANCE) # NOTE: to set weekly, need specifying a weekday
        self.universe = universe
        self.data_source = data_source
        self.alpha_model = alpha_model
        self.risk_model=risk_model
        self.data_handler = BacktestDataHandler(self.universe, data_sources=[self.data_source])

        # NOTE: ZeroFeeModel can only be used to first-test strategy -> unrealistic returns
        # fee_model = ZeroFeeModel()
        commission_percentage = kwargs.get('commission_pct', default.COMMISSION_PERCENTAGE)
        cash_buffer_percentage = kwargs.get('cash_buffer_percentage', default.CASH_BUFFER_PERCENTAGE)
        self.fee_model = PercentFeeModel(commission_pct=commission_percentage)

        backtest = BacktestTradingSession(
            self.start_dt,
            self.end_dt,
            self.universe,
            self.alpha_model,
            self.risk_model,
            signals=None,
            long_only=True,
            initial_cash=initial_cash,
            burn_in_dt=self.burn_in_dt,
            data_handler=self.data_handler,
            fee_model=self.fee_model,  # type: ignore (`qstrader` specific)
            should_rebalance_pre_market=True, # try avoid negative balance due to next-day orders execution (!)
            **kwargs
        )

        # NOTE: update the order sizer: `self.backtest.qts.portfolio_construction_model.order_sizer`
        order_sizer = CustomizedOrderSizer(backtest.broker, backtest.portfolio_id, self.data_handler, cash_buffer_percentage)
        backtest.qts.portfolio_construction_model.order_sizer = order_sizer # type: ignore (`qstrader` specific)

        self.backtest = backtest

        self.logger.debug(f'Experiment initialized')

    def run(self):
        # NOTE: 🚀 CAUTION - can take some time
        self.backtest.run()

    def plot_results(self, filename:Optional[str]=None):
        tearsheet = TearsheetStatistics(
            strategy_equity=self.backtest.get_equity_curve(),
            benchmark_equity=self.backtest.get_equity_curve(), # NOTE: no benchmark (use same)
            title='Strategy performance'
        )

        tot_returns, cagr = self.get_experiment_performance(tearsheet)
        self.logger.info(f'Strategy total return: {int(tot_returns*100)}% [CAGR: {int(cagr*100)}%]')
        print(f'Strategy total return: {int(tot_returns*100)}% [CAGR: {int(cagr*100)}%]')
        
        tearsheet.plot_results(filename)

    def get_experiment_performance(self, tearsheet:Optional[TearsheetStatistics]=None) -> Tuple:
        if tearsheet is None: tearsheet = TearsheetStatistics(
            strategy_equity=self.backtest.get_equity_curve(),
            benchmark_equity=self.backtest.get_equity_curve(), # NOTE: no benchmark (use same)
            title='Strategy performance'
        )
            
        stats = tearsheet.get_results(self.backtest.get_equity_curve())
        cum_returns = stats['cum_returns']
        tot_returns = cum_returns[-1] - 1.0
        cagr = perf.create_cagr(cum_returns, tearsheet.periods)
        return (tot_returns, cagr)