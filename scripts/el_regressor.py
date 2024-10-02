from datetime import date

import matplotlib.pyplot as plt
import numpy.random
import pandas as pd
import seaborn as sns
import torch
from torch.nn import RNN
from torch.utils.data import DataLoader

from dlstp import evaluating, setup_pytorch
from dlstp.tp2.a_data import DailyTempSeqDataset
from dlstp.tp2.c_train import train_and_early_stop
from dlstp.tp2.d_forecast import forecast

device = setup_pytorch(42)
in_seq_len = 32
batch_size = 512
out_seq_len = 128
seq_len = in_seq_len + out_seq_len + 1
target_variable = "mean_temperature"
train_dataset = DailyTempSeqDataset(
  start_date=date(1940, 1, 1),
  end_date=date(2010, 12, 31),
  seq_len=seq_len,
  target_variable=target_variable,
  device=device,
)
valid_dataset = DailyTempSeqDataset(
  start_date=date(2010, 1, 1),
  end_date=date(2019, 12, 31),
  seq_len=seq_len,
  target_variable=target_variable,
  device=device,
  target_stdscale_mean=train_dataset.target_stdscale_mean,
  target_stdscale_std=train_dataset.target_stdscale_std,
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size)
model, losses = train_and_early_stop(
  input_size=train_dataset.data.shape[-1],
  hidden_size=8,
  cell_cls=RNN,
  train_loader=train_loader,
  valid_loader=valid_loader,
  in_seq_len=in_seq_len,
  out_seq_len=out_seq_len,
  early_stopping_patience=30,
  valid_loss_eval_freq=4,
  fold_id=0,
  device=device,
)
warmup_dataset = DailyTempSeqDataset(
  start_date=date(2020, 1, 1),
  end_date=date(2020, 12, 31),
  seq_len=seq_len,
  target_variable=target_variable,
  device=device,
  target_stdscale_mean=train_dataset.target_stdscale_mean,
  target_stdscale_std=train_dataset.target_stdscale_std,
)
forecasted = forecast(
  model,
  warmup_dataset,
  n_steps=365 * 10,
  target_stdscale_mean=train_dataset.target_stdscale_mean,
  target_stdscale_std=train_dataset.target_stdscale_std,
  device=device,
)

fig, ax = plt.subplots(figsize=(5, 4))
sns.lineplot(data=pd.DataFrame(losses), x="step_id", y="loss", hue="split")
fig.tight_layout()

with evaluating(model):
  with torch.no_grad():
    batch = next(iter(valid_loader))
    fig, axes = plt.subplots(figsize=(9, 6), nrows=3, ncols=3)
    for j, i in enumerate(numpy.random.randint(low=0, high=batch.shape[0], size=9)):
      y = batch[i, :, -1].cpu().numpy()
      yh = (
        model(batch[i : i + 1, :, :], in_seq_len=in_seq_len, out_seq_len=out_seq_len)[
          0
        ][0, :, 0]
        .cpu()
        .numpy()
      )
      ax = axes[j // 3, j % 3]
      ax.plot(y, c="k")
      ax.plot(yh, c="r")
    fig.tight_layout()
