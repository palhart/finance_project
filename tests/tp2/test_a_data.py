from datetime import date, datetime

import numpy as np
import numpy.testing
import pytest
from torch.utils.data import DataLoader

from dlstp.tp2.a_data import DailyTempSeqDataset, NotEnoughDataError


def test_daily_temp_seq_dataset_init(device):
  with pytest.raises(NotEnoughDataError):
    DailyTempSeqDataset(
      start_date=date(2020, 1, 1),
      end_date=date(2020, 1, 3),
      seq_len=32,
      target_variable="mean_temperature",
      device=device,
    )
  d = DailyTempSeqDataset(
    start_date=date(2020, 1, 1),
    end_date=date(2023, 12, 31),
    seq_len=8,
    target_variable="mean_temperature",
    device=device,
  )
  assert d.n_days == 1461
  assert len(d) == 1453


def test_daily_temp_seq_dataset_data_access(device):
  d = DailyTempSeqDataset(
    start_date=date(2020, 1, 1),
    end_date=date(2023, 12, 31),
    seq_len=8,
    target_variable="max_temperature",
    device=device,
  )
  x = d[0]
  x = np.array(x.to("cpu"))
  expected_x = np.array(
    [
      [-0.5, -0.5, -0.16666666, -1.7517909],
      [-0.49726027, -0.46666667, 0.0, -1.5951859],
      [-0.49452055, -0.43333334, 0.16666669, -1.0044631],
      [-0.49178082, -0.4, 0.3333333, -1.0551294],
      [-0.4890411, -0.36666667, -0.6666667, -1.1173106],
      [-0.48630136, -0.3333333, -0.5, -1.3925208],
      [-0.48356164, -0.3, -0.3333333, -1.0850685],
      [-0.4808219, -0.26666665, -0.16666666, -1.0493717],
    ]
  )
  numpy.testing.assert_allclose(x, expected_x, atol=0.001)
  # just checking your batches are correcly constructed
  x = next(
    iter(
      DataLoader(
        DailyTempSeqDataset(
          start_date=date(2021, 1, 1),
          end_date=date(2021, 1, 31),
          seq_len=4,
          target_variable="max_temperature",
          device=device,
        ),
        batch_size=2,
        # indices of the samples that are used to construct the batches
        sampler=[2, 9],
      )
    )
  )
  x = np.array(x.to("cpu"))
  assert tuple(x.shape) == (2, 4, 4)
  expected_x = np.array(
    [
      [
        [-0.49452055, -0.43333334, -0.6666667, -0.85080326],
        [-0.49178082, -0.4, -0.5, -0.628218],
        [-0.4890411, -0.36666667, -0.3333333, -0.86525685],
        [-0.48630136, -0.3333333, -0.16666666, -0.7843168],
      ],
      [
        [-0.47534245, -0.19999999, -0.6666667, -0.75830036],
        [-0.47260273, -0.16666666, -0.5, -1.1630008],
        [-0.46986303, -0.13333333, -0.3333333, -0.3969607],
        [-0.4671233, -0.09999999, -0.16666666, 0.669714],
      ],
    ]
  )
  numpy.testing.assert_allclose(x, expected_x, atol=0.001)


def test_daily_temp_seq_dataset_fast_batch_loading(device):
  d = DailyTempSeqDataset(
    start_date=date(1945, 1, 1),
    end_date=date(1999, 12, 31),
    seq_len=128,
    target_variable="max_temperature",
    device=device,
  )
  elapsed_ms = list()
  iterator = iter(DataLoader(d, batch_size=512))
  for _ in range(10):
    start = datetime.now()
    next(iterator)
    end = datetime.now()
    elapsed_ms.append((end - start).total_seconds() * 1000)
  # I expect loading a batch should on average take less than 20 ms
  # At least, it's the case on my shitty laptop
  # (On my big machines, it takes ~600 µs)
  assert np.mean(elapsed_ms) < 20
