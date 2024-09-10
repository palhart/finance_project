import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import scipy.fftpack
from numpy.lib.stride_tricks import as_strided

DATA_DIR = Path(os.getenv("DLST_DATA_DIR", "/data/dlst"))


def load_stock_prices(symbols: list[str] | None) -> pd.DataFrame:
  path = DATA_DIR / "sp500" / "sp500_stocks.pq"
  extra_query_filter = ""
  if symbols is not None:
    assert all(isinstance(s, str) for s in symbols)
    symbolsstr = ", ".join([f"'{s}'" for s in symbols])
    extra_query_filter = f"where symbol in ({symbolsstr})"
  prices = (
    duckdb.query(
      f"""
      select date, symbol, close
      from read_parquet('{path}')
      {extra_query_filter}
      order by date
      """
    )
    .df()
    .assign(date=lambda x: pd.to_datetime(x.date))
  )
  return prices


def load_hourly_paris_temperatures() -> pd.DataFrame:
  return pd.read_parquet(DATA_DIR / "paristemp" / "hourly.pq")


def load_daily_paris_temperatures() -> pd.DataFrame:
  return pd.read_parquet(DATA_DIR / "paristemp" / "daily.pq")


def fast_autocorrelation(x: np.ndarray) -> np.ndarray:
  """
  Calculate the autocorrelation of time series x at all possible lags.

  This uses FFT to make it very efficient.

  Note: Assumes equally-spaced time points.

  """
  xp = scipy.fftpack.ifftshift((x - np.average(x)) / np.std(x))
  (n,) = xp.shape
  xp = np.r_[xp[: n // 2], np.zeros_like(xp), xp[n // 2 :]]
  f = scipy.fftpack.fft(xp)
  p = np.absolute(f) ** 2
  pi = scipy.fftpack.ifft(p)
  return np.real(pi)[: n // 2] / (np.arange(n // 2)[::-1] + n // 2)


def subsequences(arr: np.ndarray, m: int):
  """
  Return a matrix with all subsequences of length m that can be constructed from array
  arr.

  Resulting matrix shape will be len(arr) - m + 1.

  Example for arr [-4, -4, -3,  9,  0, 11,  3, -5] and m = 3:

    [-4, -4, -3]
    [-4, -3,  9]
    [-3,  9,  0]
    [ 9,  0, 11]
    [ 0, 11,  3]
    [11,  3, -5]

  """
  n = arr.size - m + 1
  s = arr.itemsize
  return as_strided(arr, shape=(m, n), strides=(s, s)).T
