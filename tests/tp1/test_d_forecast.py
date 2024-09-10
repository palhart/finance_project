import numpy as np
import numpy.testing
import pandas as pd
import pytest

from dlstp.tp1.d_forecast import fit_predict_ar1


@pytest.mark.parametrize("stock_prices", [["AAPL"]], indirect=True)
def test_fit_predict_ar1(stock_prices: pd.DataFrame):
  real = stock_prices.set_index("date").close
  predicted = fit_predict_ar1(real)
  assert isinstance(predicted.index, pd.DatetimeIndex)
  data = (
    pd.merge(
      real.rename("real").to_frame(),
      predicted.rename("predicted").to_frame(),
      how="inner",
      left_index=True,
      right_index=True,
    )
    .assign(
      real_diff_pct=lambda x: x.real.diff(1) / x.real,
      predicted_diff_pct=lambda x: x.predicted.diff(1) / x.predicted,
      residual=lambda x: x.predicted - x.real,
      residual_pct=lambda x: (x.predicted - x.real) / x.real,
      residual_diff_pct=lambda x: (x.predicted.diff(1) / x.predicted)
      - (x.real.diff(1) / x.real),
    )
    .dropna()
  )
  # predictions less than 1% of the actual price on average
  numpy.testing.assert_allclose(data.residual_pct.mean(), 0.0, atol=0.01)
  # at least 99% cross-correlation with real prices
  assert np.corrcoef(data.predicted, data.real)[0, 1] >= 0.99
  # but less than 20% cross-corr when comparing day-to-day pct differences
  assert np.corrcoef(data.predicted_diff_pct, data.real_diff_pct)[0, 1] < 0.2
