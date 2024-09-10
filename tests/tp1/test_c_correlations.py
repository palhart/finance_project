import numpy as np
import numpy.testing
import pandas as pd
import pytest

from dlstp.tp1.c_correlations import (
  compute_all_stock_cross_correlations,
  compute_monthly_stock_cross_correlations,
)


def _test_corr_matrix_properties(corrs: pd.DataFrame) -> None:
  selfcorrs = np.diagonal(corrs.values)
  numpy.testing.assert_allclose(
    selfcorrs,
    np.ones_like(selfcorrs),
    err_msg="should have a perfect 1.0 correlation with themselves",
  )
  numpy.testing.assert_allclose(
    corrs.values,
    corrs.values.T,
    err_msg="correlation matrix should be symmetrical",
  )


@pytest.mark.parametrize(
  "stock_prices", [["NVDA", "TSLA", "AAPL", "ABNB"]], indirect=True
)
def test_compute_all_stock_correlations(stock_prices: pd.DataFrame):
  corrs = compute_all_stock_cross_correlations(stock_prices)
  assert set(corrs.index.tolist()) == set(
    corrs.columns.tolist()
  ), "columns and index should be equal (all stock symbols)"
  _test_corr_matrix_properties(corrs)
  numpy.testing.assert_allclose(
    [corrs.loc["AAPL", "NVDA"], corrs.loc["TSLA", "ABNB"]],
    [0.7966734451506324, 0.44715991498731034],
  )


@pytest.mark.parametrize(
  "stock_prices", [["NVDA", "TSLA", "AAPL", "ABNB"]], indirect=True
)
def test_compute_monthly_stock_correlations(stock_prices: pd.DataFrame):
  monthly_corrs = compute_monthly_stock_cross_correlations(stock_prices)
