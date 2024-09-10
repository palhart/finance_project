from typing import cast

import numpy as np
import numpy.testing
import pandas as pd
import pytest

from dlstp.tp1.a_warmup import (
  estimate_sine_wave_params,
  generate_random_walk,
  generate_sine_wave,
  generate_whitenoise_time_series,
)


def test_generate_random_walk():
  n_steps = 50
  std = 0.3
  sampled = np.array([generate_random_walk(n_steps, std) for _ in range(10_000)])
  k = 44
  sample_var_k = sampled[:, k].std() ** 2
  theoretical_var_k = k * std**2
  numpy.testing.assert_allclose(sample_var_k, theoretical_var_k, atol=1)


@pytest.mark.parametrize("std", [3.0, 2.0, 0.1, 0.05])
def test_generate_whitenoise_time_series(std: float):
  delta_t = cast(pd.Timedelta, pd.Timedelta(milliseconds=100))
  start_dt = cast(pd.Timestamp, pd.Timestamp("2023-01-01 00:00:00"))
  end_dt = cast(pd.Timestamp, pd.Timestamp("2023-01-02 00:00:00"))
  noise_series = generate_whitenoise_time_series(std, delta_t, start_dt, end_dt)
  assert isinstance(
    noise_series.index, pd.DatetimeIndex
  ), "Index is not a DatetimeIndex"
  index_diff = noise_series.index[1] - noise_series.index[0]
  assert index_diff == delta_t, f"Index frequency is {index_diff}, expected {delta_t}"
  expected_num_samples = len(pd.date_range(start_dt, end_dt, freq=delta_t))
  assert (
    len(noise_series) == expected_num_samples
  ), f"Number of samples is {len(noise_series)}, expected {expected_num_samples}"
  mean = noise_series.mean()
  numpy.testing.assert_allclose(
    mean,
    0,
    atol=0.01,
    err_msg=f"Mean of the series is {mean}, expected to be close to 0",
  )
  std_dev = noise_series.std()
  numpy.testing.assert_allclose(
    std_dev,
    std,
    atol=0.01,
    err_msg=f"Standard deviation is {std_dev}, expected to be close to {std}",
  )


def test_generate_sine_wave():
  actual = generate_sine_wave(
    amplitude=20,
    ordinary_freq=3,  # 3 cycles per second
    duration_sec=1,  # one second sampled
    sampling_freq=16,  # 16 values per second
  )
  expected = pd.Series(
    {
      0.0: 0.0,
      0.06666666666666667: 19.02113032590307,
      0.13333333333333333: 11.755705045849465,
      0.2: -11.75570504584946,
      0.26666666666666666: -19.021130325903073,
      0.3333333333333333: -4.898587196589413e-15,
      0.4: 19.02113032590307,
      0.4666666666666667: 11.755705045849467,
      0.5333333333333333: -11.755705045849457,
      0.6: -19.021130325903076,
      0.6666666666666666: -9.797174393178826e-15,
      0.7333333333333333: 19.02113032590306,
      0.8: 11.75570504584947,
      0.8666666666666667: -11.755705045849451,
      0.9333333333333333: -19.021130325903076,
      1.0: -1.4695761589768237e-14,
    }
  )
  assert len(actual) == len(expected)
  numpy.testing.assert_allclose(
    cast(np.ndarray, actual.index.values), cast(np.ndarray, expected.index.values)
  )
  numpy.testing.assert_allclose(
    cast(np.ndarray, actual.values), cast(np.ndarray, expected.values)
  )


@pytest.mark.parametrize(
  "sine_params",
  [
    {32: 20},
    {1: 40},
    {2: 80, 40: 12},
    {64: 4, 32: 8, 16: 16},
  ],
)
def test_estimate_sine_wave_params(sine_params: dict):
  sfreq = 256
  dur_sec = 30
  signal = np.sum(
    [
      generate_sine_wave(ampl, ord_freq, dur_sec, sfreq)
      for ord_freq, ampl in sine_params.items()
    ],
    axis=0,
  )
  estimated = estimate_sine_wave_params(signal, sfreq)
  assert set(estimated.keys()) == set(sine_params.keys())
  for freq, ampl in estimated.items():
    assert np.isclose(ampl, sine_params[freq], atol=1.0)
