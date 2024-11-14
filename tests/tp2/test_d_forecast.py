import datetime

import numpy.testing
import pandas as pd
from pandas.testing import assert_series_equal
from torch.nn import RNN

from dlstp.tp2.a_data import DailyTempSeqDataset
from dlstp.tp2.b_model import ElRegressor
from dlstp.tp2.d_forecast import SequentialLoader, forecast


def test_sequential_loader(device):
  dataset = DailyTempSeqDataset(
    start_date=datetime.date(1945, 1, 1),
    end_date=datetime.date(1945, 1, 18),
    seq_len=8,
    target_variable="mean_temperature",
    device=device,
  )
  input_size = dataset.data.shape[-1]
  d = SequentialLoader(dataset)
  shapes = [tuple(x.shape) for x in d]
  assert len(shapes) == 2
  assert all(shape == (1, 8, input_size) for shape in shapes)
  dataset = DailyTempSeqDataset(
    start_date=datetime.date(1945, 1, 1),
    end_date=datetime.date(1949, 12, 31),
    seq_len=365,
    target_variable="mean_temperature",
    device=device,
  )
  d = SequentialLoader(dataset)
  shapes = [tuple(x.shape) for x in d]
  assert len(shapes) == 5
  assert all(shape == (1, 365, input_size) for shape in shapes)


def test_forecast(device):
  in_seq_len = 4
  out_seq_len = 8
  seq_len = in_seq_len + out_seq_len + 1
  target_variable = "mean_temperature"
  warmup_dataset = DailyTempSeqDataset(
    start_date=datetime.date(2020, 1, 1),
    end_date=datetime.date(2020, 12, 31),
    seq_len=seq_len,
    target_variable=target_variable,
    device=device,
  )
  model = ElRegressor(
    input_size=warmup_dataset.data.shape[-1],
    hidden_size=4,
    cell_cls=RNN,
  ).to(device)
  forecasted = forecast(
    model,
    warmup_dataset,
    n_steps=6,
    target_stdscale_mean=warmup_dataset.target_stdscale_mean,
    target_stdscale_std=warmup_dataset.target_stdscale_std,
    device=device,
  )
  expected = pd.Series(
    {
      datetime.date(2021, 1, 1): 12.798280265176146,
      datetime.date(2021, 1, 2): 13.27431806863401,
      datetime.date(2021, 1, 3): 13.918219430595641,
      datetime.date(2021, 1, 4): 12.356483903627066,
      datetime.date(2021, 1, 5): 12.177603379959578,
      datetime.date(2021, 1, 6): 12.80196991315637,
    }
  )
  assert_series_equal(forecasted, expected, atol=0.001)
