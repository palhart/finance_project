import numpy as np
import pandas as pd

from dlstp.tp1.b_autoregression import AutoregressiveModel

def convert_pct_change_to_price(pct_change: list, initial_price: float) -> list:
  price = [initial_price]
  for change in pct_change:
    next_price = price[-1] * (1 + change)
    price.append(next_price)
  return price[1:]


def predict_monthly_ari(month_prices: pd.Series, prev_month: pd.Series) -> pd.Series:
  length_month_prices = len(month_prices)
  statonary_prices = month_prices.pct_change().dropna()
  model = AutoregressiveModel()
  train_model = model.fit(data=statonary_prices.to_numpy(), order=1)
  prediction = train_model.forecast(past_observations=statonary_prices.to_numpy(), n_samples=length_month_prices)
  prediction = prediction[-length_month_prices:]
  prediction = convert_pct_change_to_price(prediction.tolist(), prev_month.iloc[-1])
  t = pd.Series(data=prediction, index=month_prices.index)
  return t

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

  prev_month = pd.Series()
  forecasted_prices = pd.Series()
  monthly_series = prices.groupby([pd.Grouper(freq='MS')])
  for month, month_prices in monthly_series:
    if prev_month.empty:
      prev_month = month_prices
      continue
    forecasted_monthly = predict_monthly_ari(month_prices, prev_month)
    if forecasted_prices.empty:
      forecasted_prices = forecasted_monthly
    else:
      forecasted_prices = pd.concat([forecasted_prices, forecasted_monthly])
    prev_month = month_prices
  return forecasted_prices




  

# Create a Series with daily data for four months
data = range(1, 122)  # Values from 1 to 121 (31 days in January + 29 in February + 31 in March + 30 in April)
index = pd.date_range(start='2020-01-01', periods=121, freq='D')  # Daily frequency for 4 months

# Create the Series
four_month_series = pd.Series(data=data, index=index)

t = fit_predict_ar1(four_month_series)


