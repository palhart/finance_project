from datetime import date, timedelta

import numpy.testing
import pandas as pd
from torch.nn import RNN
from torch.utils.data import DataLoader

from dlstp.tp2.a_data import DailyTempSeqDataset
from dlstp.tp2.b_model import ElRegressor
from dlstp.tp2.c_train import cross_val_train, evaluate, iter_cross_val_folds


def test_iter_cross_val_folds(device):
  start_date = date(1940, 1, 1)
  end_date = date(1999, 12, 31)
  in_seq_len = 256
  out_seq_len = 256
  seq_len = in_seq_len + out_seq_len + 1
  train_dataset = DailyTempSeqDataset(
    start_date=start_date,
    end_date=end_date,
    seq_len=seq_len,
    target_variable="max_temperature",
    device=device,
  )
  decade = 1940
  for train_fold, valid_fold in iter_cross_val_folds(train_dataset):
    # check that none of the training sequences are in the validation decade
    for seq_ix in train_fold.indices:
      assert (start_date + timedelta(days=seq_ix + seq_len)).year // 10 != decade // 10
    # check that all of the validation sequences are in the validation decade
    for seq_ix in valid_fold.indices:
      assert (start_date + timedelta(days=seq_ix + seq_len)).year // 10 == decade // 10
    decade += 10  # go check the next decade fold now


def test_train_el_regressor(device):
  train_output = cross_val_train(
    in_seq_len=4,
    out_seq_len=4,
    hidden_size=16,
    batch_size=128,
    early_stopping_patience=10,
    valid_loss_eval_freq=8,
    target_variable="max_temperature",
    start_date=date(2018, 11, 1),
    end_date=date(2021, 2, 1),
    cell_cls=RNN,
    device=device,
  )
  losses = pd.DataFrame(train_output["cross_val_losses"])
  assert train_output["test_loss"] < 4, "your test loss should be less than 4"
  assert set(losses.fold_id.unique()) == {0, 1}
  assert all(lowest < 4 for lowest in losses.groupby("fold_id").loss.min())


def test_evaluate(device):
  start_date = date(1990, 1, 1)
  end_date = date(1990, 2, 1)
  in_seq_len = 4
  out_seq_len = 2
  seq_len = in_seq_len + out_seq_len + 1
  dataset = DailyTempSeqDataset(
    start_date=start_date,
    end_date=end_date,
    seq_len=seq_len,
    target_variable="max_temperature",
    device=device,
  )
  loader = DataLoader(dataset, batch_size=8)
  model = ElRegressor(input_size=dataset.data.shape[-1], hidden_size=8, cell_cls=RNN)
  expected_loss = 0.7882512378692627
  loss = evaluate(model, loader, in_seq_len, out_seq_len, device)
  numpy.testing.assert_allclose(loss, expected_loss, atol=0.001)
