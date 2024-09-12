import numpy as np
import numpy.testing
import pytest

from dlstp.tp1.b_autoregression import AutoregressiveModel, NonStationaryError


def test_init():
  AutoregressiveModel()
  AutoregressiveModel(weights=np.array([0.2, 0.01, 0.25, 0.8]))


def test_forecast():
  ar = AutoregressiveModel(weights=np.array([1, 0.25, 0.7]))
  generated = ar.forecast(
    past_observations=np.array([1, 0.5]),
    n_samples=3,
  )
  expected = np.array([1, 0.5, 1.6, 2.245, 2.9715])
  numpy.testing.assert_allclose(generated, expected)


@pytest.mark.parametrize(
  "weights",
  [
    np.array([0, 0.9]),
    np.array([0.1, 0.4, 0.09]),
    np.array([1, -0.02, 0.2, -0.03]),
    np.array([-0.04, -0.84, 0.0, -0.025]),
  ],
)
def test_fit(weights):
  order = len(weights) - 1
  data = AutoregressiveModel(weights=weights).forecast(
    past_observations=np.random.randn(order),
    n_samples=300,
    noise_std=0.0001,
  )
  fitted_weights = AutoregressiveModel().fit(data, order=order).weights
  assert fitted_weights is not None
  numpy.testing.assert_allclose(fitted_weights, weights, atol=0.01)


@pytest.mark.parametrize(
  "weights",
  [
    np.array([0, -0.3]),
    np.array([0, 1.3]),
    np.array([0.1, 0.4, -0.09]),
    np.array([11, 0.4, 0.9]),
  ],
)
def test_non_stationary(weights):
  with pytest.raises(NonStationaryError):
    AutoregressiveModel(weights)
