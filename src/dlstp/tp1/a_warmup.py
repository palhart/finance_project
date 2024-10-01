"""
In this warmup you'll get to generate typical time series data and learn some basics of
the way time series are represented in pandas.
"""

from typing import cast

import numpy as np
import pandas as pd
import scipy.fft
import scipy.signal
import matplotlib.pyplot as plt

def generate_random_walk(n_steps: int, std: float) -> np.ndarray:
  """
  Generate a random walk of n_steps steps where each step adds random noise sampled from
  a normal distribution with mean 0 and standard deviation std.

  The output is a NumPy array of length n_steps containing the sampled signal.

  """
  return np.cumsum(np.random.normal(0, std, n_steps))

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
  n_steps = int(((end_dt - start_dt) / delta_t) + 1)
  return pd.Series(
    data=np.random.normal(0, std, n_steps),
    index=pd.date_range(start_dt, end_dt, freq=delta_t),
  )


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
  x = np.linspace(0, duration_sec, sampling_freq * duration_sec)
  y = amplitude * np.sin((2 * np.pi) * ordinary_freq * x)
  return pd.Series(data=y, index=x)

def estimate_sine_wave_params(signal: np.ndarray, sample_freq: int):
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
  yf = scipy.fft.rfft(signal)

 


  yf = np.array(yf)

  #absolut value of the complex number => find the module of the complex number => sqrt(real^2 + imag^2) => magnitude
  
  yf_pos = np.abs(yf) 

  yf_norm = yf_pos / (len(signal) / 2)
  
  xf = scipy.fft.rfftfreq(len(signal) , 1 / sample_freq)

  peaks, _ = scipy.signal.find_peaks(yf_pos, height=0.1)

  frequencies = xf[peaks]

  #find the amplitudes of the peaks

  amplitudes = yf_norm[peaks]

  result = dict(zip(frequencies, amplitudes))

  return result


