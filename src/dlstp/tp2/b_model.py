from typing import Type, TypeAlias

import torch
from torch import Tensor
from torch.nn import GRU, RNN, Linear, Module

CellClass: TypeAlias = Type[GRU] | Type[RNN]


class ElRegressor(Module):
  """
  Univariate autoregressive recurrent neural net.

  This model should have a single hidden layer.

  """

  def __init__(self, input_size: int, hidden_size: int, cell_cls: CellClass) -> None:
    """
    Create an autoregressive recurrent neural network.

    Parameters
    ----------
    input_size : int
      Number of dimensions of the input at any given time step.
      Since we're working on an univariate time series, this is going to be 1 plus the
      number of so-called 'time features' (exogenous variables).

    hidden_size : int
      Number of units (neurons) in the hidden layer.


    cell_cls : CellClass
      Type of cell used for the internal recurrence module (RNN or GRU).

    Tips:
      - this is where you should define the different internal submodules that are part
        of the model, i.e. the recurrent module and its dense layer to map the hidden
        state to the regressed scalar variable.
      - note that in PyTorch you can represent the data as (B, L, D) or (L, B, D), where
        B is the batch size, L is the sequence length and D is the input size. for some
        reason I ignore they decided that (L, B, D) was the default, but here we're
        working on data shaped as (B, L, D). so you should look at the `batch_first`
        argument, which should be set to True.

    """
    super().__init__()
    self.input_size = input_size
    self.cell_cls = cell_cls
    raise NotImplementedError()

  def forward(
    self,
    seq: Tensor,
    in_seq_len: int,
    out_seq_len: int,
    h0: Tensor | None = None,
  ) -> tuple[Tensor, Tensor]:
    """
    Given a sequence of length N + M + 1, where N is the in_seq_len and M is the
    out_seq_len, use the model for one-step-ahead prediction on the first N elements of
    the sequence, and then multi-step-ahead prediction on the last M elements of the
    sequence.

    Tips:
      - this is important code that you should get right, so think this through: it's
        non-trivial. use a whiteboard. sorry, this is not really a tip :)

    Parameters
    ----------
    seq : Tensor
      Batch of sequences of length in_seq_len + out_seq_len + 1.
      The shape of this tensor is (B, N + M + 1, D) where B is the batch size and D is
      the input size.

    in_seq_len : int
      Length of the sequence where the model is passed the 'true' temperature as input
      at each time step.

    out_seq_len : int
      Length of the sequence where the model is passed its own previous predicted
      temperature as input at each time step. This is used to train a multi-step-ahead
      forecasting model.

    h0 : Tensor | None
      Initial hidden state, randomly initialized if None

    Returns
    -------
    tuple[Tensor, Tensor]
      Output sequence of predictions of the target variable at each time step, and the
      last hidden state values after processing the entire sequence. Note that the shape
      of the predicted sequence is (B, N + M, 1).

      We return the last hidden state (second tensor of the tuple) because we sometimes
      need to pass it as the initial hidden state (h0) when we process contiguous
      sequences one after the other.

    """
    batch_size, seq_len, input_size = seq.size()
    assert seq_len == in_seq_len + out_seq_len + 1
    assert input_size == self.input_size
    raise NotImplementedError()
