import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dlstp import evaluating
from dlstp.tp2.a_data import DailyTempSeqDataset
from dlstp.tp2.b_model import ElRegressor
from torch.nn import RNN


class SequentialLoader(DataLoader):
  def __init__(self, dataset: DailyTempSeqDataset) -> None:
    n = len(dataset)
    seq_len = dataset.seq_len
    ixs = list(reversed(range(n, 0, -seq_len)))
    return super().__init__(
      dataset=dataset,
      sampler=ixs,
      batch_size=1,
    )


def calculate_time_features(date: datetime.date) -> torch.Tensor:
  """
  Calculate scaled time features to be passed as exogenous input variables.

  Each variable is rescaled to be in the range [-0.5, 0.5].

  For example, if the date is 2024-08-13:
    - day of year is 226, scaled to (226 - 1) / 365 - 0.5 = 0.11643835616438358
    - day of month is 13, scaled to (13 - 1) / 30 - 0.5 = -0.09999999999999998
    - day of week is 1, scaled to (1 - 1) / 6 - 0.5 = 0.0

  We're doing this because neural nets like floats centered on 0.

  We often have to use this kind of tricks when dealing with neural nets.

  """
  return torch.Tensor(
    [
      [
        [
          (date.timetuple().tm_yday - 1) / 365.0 - 0.5,
          (date.timetuple().tm_mday - 1) / 30.0 - 0.5,
          (date.timetuple().tm_wday - 1) / 6.0 - 0.5,
        ]
      ]
    ]
  )


def forecast(
  model: ElRegressor,
  warmup_dataset: DailyTempSeqDataset,
  n_steps: int,
  target_stdscale_mean: float,
  target_stdscale_std: float,
  device: str,
) -> pd.Series:
  """
  This function performs n-step-ahead forecasting using a trained ElRegressor model.

  ¿Ha aprendido nuestro modelo los patrones estacionales de los datos?

  It uses warmup data to initialize the model's hidden state, then generates predictions
  for a specified number of future time steps.

  Key aspects of the function:
    1. Uses warmup data to initialize the model's hidden state
    2. Generates predictions for n future time steps
    3. Uses each prediction as input for the next time step
    4. Applies inverse scaling to the predictions
    5. Returns a pandas Series with both warmup data and predictions

  Tips for implementation:
    1. Use the SequentialLoader to process the warmup data
    2. Set the model to evaluation mode and disable gradient computation
    3. Process the warmup data to initialize the hidden state:
      - Iterate through the warmup_data_loader
      - Use model.recurrence to update the hidden state
    4. Start the forecasting loop:
      - Initialize with the last prediction from warmup data
      - For each step:
        a. Calculate time features for the next date
        b. Combine time features with the previous prediction
        c. Use model.recurrence to get the next hidden state
        d. Use model.dense to get the next prediction
        e. Apply inverse scaling to the prediction
        f. Store the prediction with its date
    5. Create a pandas Series from the predictions that should be returned
    6. Ensure correct handling of dates throughout the process

  Remember:
    - Pay attention to tensor shapes and devices (CPU/GPU)
    - Use the provided calculate_time_features function for exogenous inputs
    - Apply the inverse scaling using target_stdscale_mean and target_stdscale_std
    - The output should include both warmup observations and new predictions

  Note that this works under the assumption that the target variable(s) are/is the same
  as the input variable(s). In other words, we can only feed as input the output of the
  model if the input contains the same variables as the output, e.g. predicting
  tomorrow's max temperature based on the previous 16 daily max temperatures.

  """
  warmup_data_loader = SequentialLoader(warmup_dataset)
  predictions = list()
  last_hidden_state = None
  with evaluating(model):
    with torch.no_grad():
      for x in warmup_data_loader:

        x.to(device)
        out , hn = model.rnn(x, last_hidden_state) # update hidden state
        last_hidden_state = hn
      
      last_prediction = model.fc(last_hidden_state)
      predictions.append(last_prediction)
      next_date = warmup_dataset.end_date
      for _ in range(n_steps):
        last_prediction = predictions[-1]
        next_date = next_date + datetime.timedelta(days=1)
        time_features = calculate_time_features(next_date).to(device)
        new_seq = torch.cat((time_features, last_prediction), dim=2)
        _ , h = model.rnn(new_seq, last_hidden_state)
        last_hidden_state = h
        new_prediction = model.fc(last_hidden_state)
        predictions.append(new_prediction)

  predictions = torch.cat(predictions, dim=1)

  predictions = predictions.squeeze()
  predictions = predictions.cpu().numpy()
  predictions = predictions * target_stdscale_std + target_stdscale_mean
  dates = pd.date_range(
    start=warmup_dataset.end_date + datetime.timedelta(days=1),
    periods=n_steps,
  ).date
  return pd.Series(predictions[1:], index=dates, dtype='float64')


