import logging
import numpy as np

from pypfopt import DiscreteAllocation

from qstrader.portcon.order_sizer.order_sizer import OrderSizer


class CustomizedOrderSizer(OrderSizer):
    """
    Creates a target portfolio with asset position quantities using the provided weights.
    """
    def __init__(
        self,
        broker,
        broker_portfolio_id,
        data_handler,
        cash_buffer_percentage=0.05
    ):
        super().__init__()

        self.broker = broker
        self.broker_portfolio_id = broker_portfolio_id
        self.data_handler = data_handler
        self.cash_buffer_percentage = self._check_set_cash_buffer(
            cash_buffer_percentage
        )
        
        self.logger = logging.getLogger('OrderSizer')
        self.logger.debug(f'Initialized CustomizedOrderSizer')

    def _check_set_cash_buffer(self, cash_buffer_percentage):
        if (
            cash_buffer_percentage < 0.0 or cash_buffer_percentage > 1.0
        ):
            raise ValueError(
                'Cash buffer percentage "%s" provided to dollar-weighted '
                'execution algorithm is negative or '
                'exceeds 100%.' % cash_buffer_percentage
            )
        else:
            return cash_buffer_percentage

    def _obtain_broker_portfolio_total_equity(self):
        return self.broker.get_portfolio_total_equity(self.broker_portfolio_id)

    def _normalise_weights(self, weights):
        if any([weight < 0.0 for weight in weights.values()]):
            raise ValueError(
                'Cash-buffered order sizing does not support '
                'negative weights. All positions must be long-only.'
            )

        weight_sum = sum(weight for weight in weights.values())

        # If the weights are very close or equal to zero then rescaling
        # is not possible, so simply return weights unscaled
        if np.isclose(weight_sum, 0.0):
            return weights
        
        # HINT: If the weights are already < 1 assume they've been scaled already
        # However, return only those with weights > 0
        return {
                asset: weight
                for asset, weight in weights.items()
                if weight > 0
        } if weight_sum <= 1 else {
            asset: (weight / weight_sum)
            for asset, weight in weights.items()
            if weight > 0
        }

    def __call__(self, dt, weights):

        total_equity = self._obtain_broker_portfolio_total_equity()
        cash_buffered_total_equity = total_equity * (
            1.0 - self.cash_buffer_percentage
        )

        self.logger.debug(f'({dt}) CustomizedOrderSizer: before sizing: {str(weights)}. |Balance: {cash_buffered_total_equity}')


        # Pre-cost dollar weight
        N = len(weights)
        if N == 0:
            # No forecasts so portfolio remains in cash
            # or is fully liquidated
            return {}

        # Ensure weight vector sums to unity
        normalised_weights = self._normalise_weights(weights)
        self.logger.debug(f'({dt}) CustomizedOrderSizer: normalised_weights: {str(normalised_weights)}. |Balance: {cash_buffered_total_equity}')

        weight_sum = sum(weight for weight in normalised_weights.values())

        # If the weights are very close or equal to zero
        # then return an empty portfolio
        # NOTE: when portfolio has some assets, must provide 0 as values
        if np.isclose(weight_sum, 0.0):
            target_portfolio = {} 
            for asset in normalised_weights.keys():
                target_portfolio[asset] = {"quantity": 0}
            return target_portfolio

        ################################################################################################
        ###     DiscreteAllocation 
        assets = list(normalised_weights.keys())
        ## self.logger.info(f'About to retrieve assets prices {assets} ({type(assets)})')
        prices = self.data_handler.data_sources[0].get_prices_on_date(dt, assets)
        
        self.logger.info(f'({dt}) CustomizedOrderSizer: DiscreteAllocation: {str(normalised_weights)}.')
        discrete_allocation = DiscreteAllocation(normalised_weights, prices, total_portfolio_value=int(cash_buffered_total_equity))
        portfolio_dict, leftover = discrete_allocation.greedy_portfolio() # lp_portfolio()
        self.logger.info(f'({dt}) CustomizedOrderSizer: calculated portfolio: {str(portfolio_dict)}. |Balance: {cash_buffered_total_equity} |Leftover: {leftover}')

        ## convert to structure
        target_portfolio = {}
        for asset, asset_quantity in sorted(portfolio_dict.items()):
            target_portfolio[asset] = {"quantity": asset_quantity}
        
        nullify_assets = (asset for asset in weights.keys() if asset not in target_portfolio.keys())
        for asset in nullify_assets:
            self.logger.debug(f'({dt}) CustomizedOrderSizer: portfolio: liquidate {asset} position')
            target_portfolio[asset] = {"quantity": 0}

        self.logger.debug(f'({dt}) CustomizedOrderSizer: TARGET portfolio: {str(target_portfolio)}.')
        return target_portfolio