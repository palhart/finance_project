import pandas as pd
import pytest

from dlstp import load_stock_prices, setup_pytorch


@pytest.fixture(scope="session")
def stock_prices(request) -> pd.DataFrame:
  symbols = list()
  if request.param is not None:
    symbols = request.param
    assert isinstance(symbols, list)
    assert all(isinstance(s, str) for s in symbols)
  return load_stock_prices(symbols)


@pytest.fixture(scope="function")
def device() -> str:
  # set up numpy and torch random seeds to 42 and detect gpus
  # this is done for each test to ensure reproducibility in-between test runs
  return setup_pytorch(42)
