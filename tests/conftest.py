import pandas as pd
import pytest

from dlstp import load_stock_prices


@pytest.fixture(scope="session")
def stock_prices(request) -> pd.DataFrame:
  symbols = list()
  if request.param is not None:
    symbols = request.param
    assert isinstance(symbols, list)
    assert all(isinstance(s, str) for s in symbols)
  return load_stock_prices(symbols)
