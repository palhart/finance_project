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
    self.weights = weights
    raise NotImplementedError()

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
    raise NotImplementedError()

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
    raise NotImplementedError()
