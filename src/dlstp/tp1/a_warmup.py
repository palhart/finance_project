"""
In this warmup you'll get to generate typical time series data and learn some basics of
the way time series are represented in pandas.
"""

from typing import cast

import numpy as np
import pandas as pd
import scipy.fft
import scipy.signal


def generate_random_walk(n_steps: int, std: float) -> np.ndarray:
  """
  Generate a random walk of n_steps steps where each step adds random noise sampled from
  a normal distribution with mean 0 and standard deviation std.

  The output is a NumPy array of length n_steps containing the sampled signal.

  """
  raise NotImplementedError()


def generate_whitenoise_time_series(
  std: float,
  delta_t: pd.Timedelta,
  start_dt: pd.Timestamp,
  end_dt: pd.Timestamp,
) -> pd.Series:
  """
  Generate a white noise time series at a given sampling frequency and with a given
  standard deviation (std).

  The resulting pandas Series should have a DatetimeIndex and the values should follow
  a normal distribution with mean 0 and the given std.

  Tips:
      - use pd.date_range to generate the index
      - use numpy's random module to generate random numbers

  """
  raise NotImplementedError()


def generate_sine_wave(
  amplitude: float,
  ordinary_freq: int,
  duration_sec: int,
  sampling_freq: int,
) -> pd.Series:
  """
  Generate a sine wave with the given amplitude and ordinary frequency.

  The resulting pd.Series should be indexed by the elapsed time as a floating-point
  number of seconds up until duration_sec seconds (included).

  Note:
      - ordinary_freq is the nb of oscillations/cycles per sec
      - sampling_freq is the nb of sampled values per second
      - assume the phase to be 0 (first value should be zero)

  """
  raise NotImplementedError()


def estimate_sine_wave_params(signal: np.ndarray, sample_freq: int) -> dict:
  """
  Given a time series that is a composite of multiple sine waves, find the frequencies
  and amplitudes of the original sine waves.

  The result should be a dict[int, float] mapping the ordinary frequencies to their
  amplitudes. The number of key-values in the resulting dict should correspond to the
  number of sine waves composing the given signal.

  Tips:
      - use scipy.fft to move the signal from the time domain to the frequency domain
      - use scipy.signal.find_peaks to find the peak amplitudes

  """
  raise NotImplementedError()
