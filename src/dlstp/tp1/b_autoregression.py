import numpy as np

from dlstp import subsequences


class NonStationaryError(ValueError):
  pass


class AutoregressiveModel:
  def __init__(self, weights: np.ndarray | None = None) -> None:
    """
    If some weights are passed and the underlying AR process is of order 1 or 2, check
    that the weights respect the stationarity condition of AR(1) and AR(2) processes.

    You should raise a NonStationaryError if the condition is violated.

    """
    if weights is not None:
      if len(weights) == 2:
        if np.abs(weights[1]) >= 1:
          raise NonStationaryError("AR(1) weights must be less than 1 in absolute value")
      elif len(weights) == 3:
        if np.abs(weights[1] + weights[2]) >= 1 or not (0 < weights[1] < 1):
          raise NonStationaryError("AR(2) weights must sum to less than 1 in absolute value")
    self.weights = weights

  @property
  def order(self) -> int:
    """
    Returns the order p of the AR(p) model.

    Note: we substract one because of the extra bias weight in the model.
    """
    if self.weights is None:
      raise ValueError("Cannot get order of model with no weights")
    return len(self.weights) - 1

  def forecast(
    self,
    past_observations: np.ndarray,
    n_samples: int,
    noise_std: float | None = None,
  ) -> np.ndarray:
    

    

    """
    Given an initialized model with non-None weights and past observations of a time
    series, use the model to sample n_samples future data points, possibly with noise.

    If noise_std is None then the error term should be zero at every step.

    Otherwise, the noise should be normally distributed with mean 0 and standard
    deviation noise_std.

    The output is a 1-D NumPy array with both past_observations and the n_samples
    generated samples concatenated.

    """


    if self.weights is None:
      raise ValueError("Cannot forecast from a model with no weights")
    order = len(self.weights) - 1
    if len(past_observations) < order:
      raise ValueError(f"Cannot forecast without at least {order} past values")
    result = list(past_observations)
    for _ in range(n_samples):
      prediction = self.weights[0] 
      for i in range(1, len(self.weights)):
        prediction += self.weights[-i] * result[-i]
      if noise_std is not None:
        prediction += np.random.normal(0, noise_std)
      result.append(prediction)
    return np.array(result)




  def fit(self, data: np.ndarray, order: int) -> "AutoregressiveModel":
    """
    Fit the autoregressive model's weights based on the given data.

    First construct all of `data`'s possible subsequences of length order + 1 and
    fit a linear regression model on these sequences.

    Tip:
        - Use np.linalg.lstsq to obtain a numerically stable solution
        - You can use the function dlstp.subsequences provided to you to generate the
          subsequences

    """


    assert self.weights is None
    subsequences_ = subsequences(data, order + 1)
    X = subsequences_[:, :-1]
    y = subsequences_[:, -1]
    
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    weights = np.linalg.lstsq(X, y, rcond=None)[0]


    return AutoregressiveModel(weights)
  


  


    