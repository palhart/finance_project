import contextlib
import os
import random
from datetime import date
from pathlib import Path
from typing import Generator, Literal, TypeAlias, cast

import duckdb
import numpy as np
import pandas as pd
import scipy.fftpack
import torch
import torch.backends.cudnn
import torch.nn
from loguru import logger
from numpy.lib.stride_tricks import as_strided

DATA_DIR = Path(os.getenv("DLST_DATA_DIR", "/data/dlst"))
if not DATA_DIR.is_dir():
  raise FileNotFoundError(
    f"You specified a different data dir but it is missing: {DATA_DIR}"
  )


# possible variable names in the Paris temperature dataset
TempVarName: TypeAlias = (
  Literal["min_temperature"] | Literal["mean_temperature"] | Literal["max_temperature"]
)


@contextlib.contextmanager
def evaluating(model: torch.nn.Module) -> Generator[torch.nn.Module, None, None]:
  """
  Helper context to set the model in evaluation mode temporarily.

  Often used in conjunction with the torch.no_grad() context.

  Example usage:

  with evaluating(my_model):
    with torch.no_grad():
      # some code that does not need to calculate gradients

  """
  is_train = model.training
  try:
    model.eval()
    yield model
  finally:
    if is_train:
      model.train()


def setup_pytorch(random_seed: int | None = None) -> str:
  torch.multiprocessing.set_sharing_strategy("file_system")
  if torch.cuda.is_available():
    device = f"cuda:{torch.cuda.current_device()}"
    device_name = torch.cuda.get_device_name(device)
    torch.backends.cudnn.benchmark = True
    logger.info(f"using cuda device {device_name}")
  else:
    device = "cpu"
    logger.warning("no CUDA GPU available. using cpu instead")
  if random_seed is not None:
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if device.startswith("cuda"):
      torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
  return device


def count_model_params(model: torch.nn.Module) -> int:
  return sum(p.numel() for p in model.parameters())


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


def load_daily_paris_temperatures(
  start_date: date | None = None,
  end_date: date | None = None,
) -> pd.DataFrame:
  daily_temp_path = DATA_DIR / "paristemp" / "daily.pq"
  query = f"""
  select
    datetime,
    min_temperature::FLOAT as min_temperature,
    mean_temperature::FLOAT as mean_temperature,
    max_temperature::FLOAT as max_temperature
  from read_parquet('{daily_temp_path}')
  """
  if start_date is not None or end_date is not None:
    query += """
    where
    """
    conditions = list()
    if start_date is not None:
      conditions.append(f"datetime >= '{start_date}'")
    if end_date is not None:
      conditions.append(f"datetime <= '{end_date}'")
    query += " and ".join(conditions)
  data = duckdb.sql(query).df()
  return data


def get_daily_paris_temp_dataset_date_boundaries():
  """Get starting and ending date in daily paris temperature dataset"""
  daily_temp_path = DATA_DIR / "paristemp" / "daily.pq"
  start_date, end_date = cast(
    tuple[date, date],
    duckdb.sql(
      f"""
      select
        min(datetime)::DATE as start_date,
        max(datetime)::DATE as end_date
      from read_parquet('{daily_temp_path}')
      """
    ).fetchone(),
  )
  return start_date, end_date


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
