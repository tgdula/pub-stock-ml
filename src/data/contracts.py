from types import SimpleNamespace

_columns_dict = {
    'OPEN': 'open',
    'HIGH': 'high',
    'LOW': 'low',
    'CLOSE': 'close',
    'PRICE': 'price',
    'VOLUME': 'volume',
    'DATE': 'date',
    'STOCK': 'stock',
    'RETURN': 'return_1d',
    'RETURN5': 'return_5d',
    'SCORE': 'return_5d_predicted',
}

Columns = SimpleNamespace(**_columns_dict) # NOTE: name's capitalized: (1) to not confuse with variables, (2) was a class..