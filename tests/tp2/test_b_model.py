import torch
from numpy.testing import assert_allclose
from torch import Tensor
from torch.nn import GRU, RNN

from dlstp import count_model_params, evaluating
from dlstp.tp2.b_model import ElRegressor


def test_el_regressor_basic_rnn(device):
  """
  Note that this test is reproducible even though the model parameters are randomly
  initialized because we set the random seed to 42 at the start of every test.

  This means that your computation should result in exactly the same floats as my
  implementation.

  """
  in_seq_len = 2
  out_seq_len = 4
  model = ElRegressor(
    input_size=1,
    hidden_size=2,
    cell_cls=RNN,
  ).to(device)
  assert count_model_params(model) == 13
  # a batch with two dummy sequences
  x = Tensor(
    [
      [[0.7], [0.3], [0.6], [0.9], [0.2], [0.3], [0.99]],
      [[0.2], [0.1], [1.7], [0.0], [0.2], [-0.7], [1.7]],
    ],
  ).to(device)
  batch_size, _, input_size = x.shape
  with evaluating(model):
    with torch.no_grad():
      yh = model(x, in_seq_len, out_seq_len)[0].cpu().numpy()
  assert yh.shape == (batch_size, in_seq_len + out_seq_len, input_size)
  expected_yh = [
    [[0.91631], [0.82850647], [0.9185386], [1.0039966], [1.0295002], [1.0395176]],
    [[0.7510188], [0.6775626], [0.85407674], [0.97948754], [1.0204011], [1.0362302]],
  ]
  assert_allclose(yh, expected_yh, atol=0.0001)


def test_el_regressor_gru(device):
  in_seq_len = 2
  out_seq_len = 4
  model = ElRegressor(
    input_size=1,
    hidden_size=2,
    cell_cls=GRU,
  ).to(device)
  assert count_model_params(model) == 33
  # a batch with two dummy sequences
  x = Tensor(
    [
      [[0.7], [0.3], [0.6], [0.9], [0.2], [0.3], [0.99]],
      [[0.2], [0.1], [1.7], [0.0], [0.2], [-0.7], [1.7]],
    ],
  ).to(device)
  batch_size, _, input_size = x.shape
  with evaluating(model):
    with torch.no_grad():
      yh = model(x, in_seq_len, out_seq_len)[0].cpu().numpy()
  assert yh.shape == (batch_size, in_seq_len + out_seq_len, input_size)
  expected_yh = [
    [[0.3445592], [0.28614563], [0.2640866], [0.25718156], [0.25507292], [0.25444737]],
    [[0.37436217], [0.31013465], [0.270477], [0.25850928], [0.2551455], [0.2543013]],
  ]
  assert_allclose(yh, expected_yh, atol=0.0001)
