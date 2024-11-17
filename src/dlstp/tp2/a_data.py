"""
This module defines the daily temperature dataset.

Most of the code for loading the data is provided to you.

A few methods remain to be implemented. See the `NotImplementedError`s below.

"""

from datetime import date

import duckdb
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

import dlstp
from dlstp import TempVarName


class NotEnoughDataError(ValueError):
  pass


class DailyTempSeqDataset(Dataset):
  """
  Dataset of daily temperature sequences.

  This dataset contains all sub-sequences of length seq_len that can be constructed
  from daily temperatures measured between start_date and end_date.

  The target variable is the mean, max or min temperature in the data.

  Optional mean and standard deviation parameters are passed to rescale the target
  variable.

  Some (rescaled) exogenous time features are calculated based on the day of the year,
  day of the month and day of the week.

  """

  def __init__(
    self,
    start_date: date,
    end_date: date,
    seq_len: int,
    target_variable: TempVarName,
    device: str,
    target_stdscale_mean: float | None = None,
    target_stdscale_std: float | None = None,
  ) -> None:
    """
    Create a dataset of daily temperature sequences.

    Parameters
    ----------
    start_date : date
      Date of the first observation in the data.

    end_date : date
      Date of the last observation in the data.

    seq_len : int
      Length of the sequences that should be generated.

    target_variable : TempVarName
      Variable that should be auto-regressed (i.e. it's both the input and output).

    device : str
      Device on which the data should be put (a cpu or gpu).

    target_stdscale_mean : float | None
      If provided, used for standard-scaling the target variable.
      Otherwise, will be estimated from the loaded data.

    target_stdscale_std : float | None
      If provided, used for standard-scaling the target variable.
      Otherwise, will be estimated from the loaded data.

    """
    super().__init__()
    self.start_date = start_date
    self.end_date = end_date
    self.seq_len = seq_len
    self.target_variable = target_variable
    self._target_stdscale_mean = target_stdscale_mean
    self._target_stdscale_std = target_stdscale_std
    self.device = device
    self._data_tensor = None
    if self.n_days < self.seq_len:
      raise NotEnoughDataError(
        f"{self.n_days} observations in the data, "
        f"which is not sufficient to construct a sequence of length {self.seq_len}"
      )

  @property
  def n_days(self) -> int:
    """
    Returns the number of days in the dataset

    Tips
      - do NOT load the data: you can calculate that simply from the start/end dates

    """
    return (self.end_date - self.start_date).days

  def __getitem__(self, index: int) -> Tensor:
    """
    Returns the subsequence at a given index.

    If the index is 0, it's the sequence that starts at date `start_date`, etc.

    """
    return self.__getitems__([index])[0]

  def __getitems__(self, indices: list) -> list[Tensor]:
    """
    Given a list of indices of sub-sequences in the dataset, returns the corresponding
    sub-sequences (actual data) as a list of tensors.

    This function exists to make it more efficient to load B sub-sequences using clever
    indexing, rather than loading each subsequence one by one (with __getitem__).

    Show me your indexing/slicing skills!

    """
    return [self.data[i : i + self.seq_len] for i in indices]

  def __len__(self) -> int:
    """
    Returns the number of sub-sequences of length seq_len that can be constructed
    from the time series of daily temperatures between start_date and end_date.

    Tips
      - this is a very simple mathematical formula

    """
    return self.n_days - self.seq_len + 1

  @property
  def target_stdscale_mean(self) -> float:
    if self._target_stdscale_mean is None:
      self.data
      assert self._target_stdscale_mean is not None
    return self._target_stdscale_mean

  @property
  def target_stdscale_std(self) -> float:
    if self._target_stdscale_std is None:
      self.data
      assert self._target_stdscale_std is not None
    return self._target_stdscale_std

  @property
  def data(self) -> Tensor:
    """
    Lazy loading the actual data into a PyTorch tensor.

    The data has a shape (N, D), where N is the total number of observations in the time
    series (number of days in the data) and where D is the number of variables observed
    at each time step (including the target variable and time features).

    Time features are rescaled to be in [-0.5, 0.5] (see code below) and include day of
    the year, day of the month and day of the week.

    These time features are useful for the model to know "what day it is" and are
    considered exogenous variables.

    """
    if self._data_tensor is None:  # first time loading
      daily_temp_path = dlstp.DATA_DIR / "paristemp" / "daily.pq"
      data = duckdb.sql(
        f"""
          select
            datetime::DATE as date,
            dayofyear(datetime)::FLOAT as day_of_year,
            dayofmonth(datetime)::FLOAT as day_of_month,
            dayofweek(datetime)::FLOAT as day_of_week,
            min_temperature::FLOAT as min_temperature,
            mean_temperature::FLOAT as mean_temperature,
            max_temperature::FLOAT as max_temperature
          from read_parquet('{daily_temp_path}')
          where
            datetime >= '{self.start_date}'
            and datetime <= '{self.end_date}'
          order by datetime
          """
      ).df()
      if self.target_variable not in data.columns:
        raise ValueError(f"Unknown variable {self.target_variable}")
      # calculate standard scaling params if not given
      if self._target_stdscale_mean is None or self._target_stdscale_std is None:
        self._target_stdscale_mean = float(data[self.target_variable].mean())
        self._target_stdscale_std = float(data[self.target_variable].std())
      data[self.target_variable] = (
        data[self.target_variable] - self._target_stdscale_mean
      ) / self._target_stdscale_std
      data["day_of_year"] = (data["day_of_year"] - 1) / 365.0 - 0.5
      data["day_of_month"] = (data["day_of_month"] - 1) / 30.0 - 0.5
      data["day_of_week"] = (data["day_of_week"] - 1) / 6 - 0.5
      sorted_cols = ["day_of_year", "day_of_month", "day_of_week", self.target_variable]
      self._data_tensor = torch.from_numpy(data[sorted_cols].values).to(
        device=self.device
      )
    return self._data_tensor  # return pre-loaded in-memory data
