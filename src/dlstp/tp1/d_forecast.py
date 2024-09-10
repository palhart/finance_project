import numpy as np
import pandas as pd

from dlstp.tp1.b_autoregression import AutoregressiveModel


def fit_predict_ar1(prices: pd.Series) -> pd.Series:
  """
  You are a young student who thinks he's gonna make some good money on the market
  against the big guys from Wall Street. Your last idea is to use an autoregressive
  model AR(1) to forecast future price movements.

  In this function, your goal is to 'backtest' an autoregressive model on historical
  data that's gonna be re-trained each month on the previous month. That is, you don't
  train on any data older than a month because you want your model to capture the most
  recent dynamics of the stock price.

  Each first day of the month, the model is trained on the previous month's data, and a
  price forecast is generated for the current month up until the last day of the month
  (included). Repeat that until you reach the end of the time series.

  If some days are missing in the data, you should reindex and return a prices series
  with all dates between the starting date and ending date, forward filling the prices.

  Tips:
    - Obviously you can't predict anything on the first month of the time series because
      no model has been fitted yet. The first month should therefore not be part of your
      output pd.Series of predictions
    - Remember what we said in class about how to use differencing to make a time series
      stationary. Your AR model should not be predicting the raw price but the % change
      (positive or negative) in the price the next day
    - Careful about future information leakage (in the input of the model)

  """
  raise NotImplementedError()
