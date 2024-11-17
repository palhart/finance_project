import collections
from datetime import date, timedelta
from io import BytesIO
from typing import Iterator, Literal, Sequence, TypedDict
import pandas as pd
from torch.nn import RNN
import torch
from torch.nn import MSELoss
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, Subset

import dlstp
from dlstp import TempVarName
from dlstp.tp2.a_data import DailyTempSeqDataset
from dlstp.tp2.b_model import CellClass, ElRegressor


class MeasuredLoss(TypedDict):
  """
  Used to represent a measured loss during training.

  We used a TypedDict for static typing purposes.

  """

  fold_id: int  # unique identifier of the cross-validation fold
  step_id: int  # number incremented to uniquely identify each gradient descent step
  epoch_id: int  # each full pass on the training data
  batch_id: int  # incremented identifier of the batch within each epoch
  # whether the loss is calculated on the train, valid or test split of the data
  split: Literal["train"] | Literal["valid"] | Literal["test"]
  # the actual measured loss value (mean)
  loss: float


class TrainingOutput(TypedDict):
  # model that obtained the lowest validation loss during cross-validation
  best_cross_val_model: ElRegressor
  # losses of each cross validation fold
  cross_val_losses: Sequence[MeasuredLoss]
  # loss measured on the test set
  test_loss: float


def save_model_to_memory(model: ElRegressor) -> BytesIO:
  """
  Helper function used to save a model in-memory (instead of a file).

  Useful to keep track of the model with the lowest validation loss, which should be the
  one returned during cross validation.

  """
  buffer = BytesIO()
  torch.save(model.state_dict(), buffer)
  return buffer


def load_model_from_memory(
  input_size: int,
  hidden_size: int,
  cell_cls: CellClass,
  buffer: BytesIO,
) -> ElRegressor:
  """Same but loading the model back from in-memory bytes."""
  buffer.seek(0)
  model = ElRegressor(input_size=input_size, hidden_size=hidden_size, cell_cls=cell_cls)
  model.load_state_dict(torch.load(buffer, weights_only=True))
  return model


def iter_cross_val_folds(
  train_dataset: DailyTempSeqDataset,
) -> Iterator[tuple[Subset[DailyTempSeqDataset], Subset[DailyTempSeqDataset]]]:
  """
  Generator which yields k cross-validation folds of a given training dataset.

  This yields tuples (train_fold, valid_fold) where train_fold is actually used for
  training the model (through mini-batch gradient descent) and where valid_fold is used
  for evaluating the model and early stopping.

  Because we have sequential data, we can't simply consider each sequence individually.
  We will use decades to split our data instead. Each decade is a fold, and we employ a
  leave-one-out strategy where one decade is kept for validation, and the other decades
  are used for training.

  The number of folds is going to be equal to the number of decades in the training
  data. Folds should be yielded sequentially, ordered by the decade used as the
  validation fold. In other words, the first yielded pair should have the 1940 decade as
  its validation fold (second tuple value).

  """    
  start_date: date = train_dataset.start_date
  end_date: date = train_dataset.end_date
  seq_len = train_dataset.seq_len

  decade_indices = {}

  for i in range(len(train_dataset)):
    sequence_start_date = start_date + timedelta(days=i + seq_len)
    decade = (sequence_start_date.year // 10) * 10

    if decade not in decade_indices:
      decade_indices[decade] = []
    
    decade_indices[decade].append(i)

  decades = sorted(decade_indices.keys())

  for decade in decades:
    training_decades = [d for d in decades if d != decade]

    training_indices = []
    for d in training_decades:
      training_indices.extend(decade_indices[d])

    validation_indices = decade_indices[decade]

    train_fold = Subset(train_dataset, training_indices)
    valid_fold = Subset(train_dataset, validation_indices)

    yield train_fold, valid_fold

def evaluate(
  model: ElRegressor, loader: DataLoader, in_seq_len: int, out_seq_len: int, device: str
) -> float:
  """
  This function evaluates a trained ElRegressor model on a given dataset.

  It calculates the average loss (MSELoss) over all samples in the provided data loader.

  Key aspects of the function:
    1. Sets the model to evaluation mode
    2. Disables gradient computation for efficiency
    3. Iterates through the data loader
    4. Performs forward passes on the model
    5. Calculates the total loss and number of samples
    6. Computes and returns the mean loss

  Tips for implementation:
    1. Use torch.no_grad() to disable gradient computation during evaluation
    2. Set the model to evaluation mode using model.eval() or the `evaluating` context
       manager
    3. Initialize variables to keep track of total loss and total number of samples
    4. Iterate through the data loader, moving input data to the specified device
    5. Perform a forward pass on the model for each batch
    6. Calculate the loss using MSELoss with reduction="sum" for accurate mean
       calculation
    7. Accumulate the loss and count the number of samples
    8. After processing all batches, calculate the mean loss
    9. Remember to handle the device (CPU/GPU) correctly for input data and model

  Note: This function is crucial for monitoring model performance during training and
  for final model evaluation. Ensure it's implemented correctly for reliable results.

  Returns
  -------
  float
    Average loss over the given data samples.

  """
  model.to(device)
  model.eval()

  criterion = MSELoss(reduction='sum')
  total_loss = 0 
  sample_size = 0
  with torch.no_grad():
    for seq in loader:

      seq = seq.to(device = device)


      output_seq, _ = model.forward(seq, in_seq_len, out_seq_len)

      eval_seq = seq[:, 1:, -1].unsqueeze(-1)

      sample_size += eval_seq.size(0) * eval_seq.size(1)


      loss = criterion(output_seq, eval_seq)


      total_loss += loss.item()
    

  return total_loss / sample_size



def train_and_early_stop(
  input_size: int,
  hidden_size: int,
  cell_cls: CellClass,
  train_loader: DataLoader,
  valid_loader: DataLoader,
  in_seq_len: int,
  out_seq_len: int,
  early_stopping_patience: int,
  valid_loss_eval_freq: int,
  fold_id: int,
  device: str,
) -> tuple[ElRegressor, list[MeasuredLoss]]:
  """
  This function implements a training loop with early stopping.

  The function trains an ElRegressor model on temperature data, evaluating its
  performance on a validation set at regular intervals. Training stops when the
  validation loss doesn't improve for a specified number of steps.

  Training neural networks can be tricky, they can easily fall into local minima. You
  can play around with the `weight_decay` parameter of the Adam optimizer to add some
  regularization to the loss function, and you can also play with the learning rate (a
  smaller learning rate will make the training slower but more stable), the batch size,
  and the valid_loss_eval_freq parameter (decreasing the frequency will lead to slower
  training but better monitoring of overfitting).

  Key aspects of the function:
    1. Initializes the model, Adam optimizer, and MSELoss loss function
    2. Implements a training loop that processes batches of data
    3. Periodically evaluates the model on a validation set
    4. Implements early stopping based on validation loss
    5. Keeps track of the best model (lowest validation loss)
    6. Returns the best model and a list of measured losses

  Tips for implementation:
    1. Start by initializing the model, Adam optimizer, and loss function
    2. Set up variables to track the best model, lowest validation loss, and early
       stopping criteria
    3. Implement nested loops: outer loop for epochs, inner loop for batches
    4. Use the train_loader to get batches of training data
    5. Implement the forward pass, loss calculation, and backward pass for each batch
    6. Periodically evaluate the model on the validation set using the valid_loader and
       the evaluate function above
    7. Update the best model and reset the early stopping counter if a new lowest
       validation loss is found
    8. Increment the early stopping counter and check if training should stop
    9. Keep track of losses for both training and validation in the losses list
    10. After training, load the best model and return it along with the list of losses

  Remember to handle the device (CPU/GPU) correctly and use the helper functions like
  save_model_to_memory, load_model_from_memory, and evaluate.

  Parameters
  ----------
  input_size : int
    Number of dimensions of the input at any given time step.
    Since we're working on an univariate time series, this is going to be 1 plus the
    number of so-called 'time features' (exogenous variables).

  hidden_size : int
    Number of neurons in the hidden layer of the model.

  cell_cls : CellClass
    Type of cell in the recurrent part of the model.

  train_loader : DataLoader
    Loader of shuffled batches of sequences used for fitting the model through
    mini-batch gradient descent.

  valid_loader : DataLoader
    Loader of batches of sequences used for evaluating the model and early stopping when
    the validation loss is not decreasing.

  in_seq_len : int
    Length of the sequences used as input of the model.

  out_seq_len : int
    Length of the sequences the model should be trained to forecast.

  early_stopping_patience : int
    Number of steps until we stop training because the validation loss did not decrease
    (cumulative number of steps).

  valid_loss_eval_freq : int
    Frequency (in number of training batches) at which the validation loss is evaluated.
    Calculating the validation loss at every batch would be creazy, and therefore we
    only calculate it at some frequency.

  fold_id : int
    Unique identifier of the cros-validation fold, which we need to keep track of in the
    MeasuredLoss dicts, which will go in our TrainingOutput dict.

  device : str
    Device identifier, e.g. 'cpu' or 'cuda:0'.

  Returns
  -------
  list of MeasuredLoss
    List of losses calculated (both training and validation losses).

  """

  model = ElRegressor(input_size, hidden_size, cell_cls).to(device)

  optimizer = Adam(model.parameters())
  criterion = MSELoss(reduce='sum')

  best_model = model
  early_stopping_counter: int = 0
  losses : list[MeasuredLoss] = []
  lowest_validation_loss : float = 100.0
  batch_counter : int = 0 
  while early_stopping_counter < early_stopping_patience:

    for seq in train_loader:
      model.train()
      batch_counter += 1 
      seq.to(device)

      output,_ = model(seq, in_seq_len, out_seq_len)

      eval_seq = seq[:, 1:, -1].unsqueeze(-1)

      loss = criterion(eval_seq, output)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      step_id = losses[-1]['step_id'] if len(losses) > 0 else 0

      loss_mesure = MeasuredLoss(
          fold_id=fold_id,
          step_id=step_id + 1,
          epoch_id=early_stopping_counter,  # You need to set the correct epoch_id
          batch_id=batch_counter,
          split="train",
          loss=loss.item()
      )

      losses.append(loss_mesure)



      if batch_counter % valid_loss_eval_freq == 0:
        loss_eval = evaluate(model=model, loader=valid_loader, in_seq_len=in_seq_len,
                        out_seq_len=out_seq_len,device=device)
        if lowest_validation_loss > loss_eval:
          early_stopping_counter = 0
          best_model = model
          lowest_validation_loss = loss_eval
          
        else:
          early_stopping_counter += 1
        loss_mesure = MeasuredLoss(
            fold_id=fold_id,
            step_id=step_id + 1,
            epoch_id=early_stopping_counter,  # You need to set the correct epoch_id
            batch_id=batch_counter,
            split="valid",
            loss=loss_eval
        )
        losses.append(loss_mesure)
      
  return best_model, losses
      


      


def cross_val_train(
  target_variable: TempVarName,
  in_seq_len: int,
  out_seq_len: int,
  hidden_size: int,
  cell_cls: CellClass,
  batch_size: int,
  early_stopping_patience: int,
  valid_loss_eval_freq: int,
  device: str,
  start_date: date | None = None,
  end_date: date | None = None,
) -> TrainingOutput:
  """
  This function performs cross-validation training of an ElRegressor model.

  It trains the model using k-fold cross-validation, selects the best model based on
  validation loss, and evaluates the final model on a test set.

  The last third of the data is kept for testing. Training is done on first 2 thirds.

  Key aspects of the function:
    1. Prepares the dataset, splitting it into training and test sets
    2. Performs k-fold cross-validation using the iter_cross_val_folds function to
       define the decade-based folds of the training dataset
    3. Trains the model for each fold using train_and_early_stop
    4. Keeps track of the best model across all folds
    5. Evaluates the best model on the test set
    6. Returns the best model, cross-validation losses, and test loss

  Tips for implementation:
    1. Use the helper functions (DailyTempSeqDataset, iter_cross_val_folds,
       train_and_early_stop, evaluate) that you have implemented
    2. Calculate the split_date for separating training and test data, keep the last
       (temporaly) third of the data for the test set, train on the first 2/3 of data
    3. Create train_dataset and test_dataset using DailyTempSeqDataset
    4. Implement the cross-validation loop:
       - Use iter_cross_val_folds to get train and validation subsets for each fold
       - Create DataLoaders for train and validation subsets
       - Call train_and_early_stop for each fold
       - Keep track of the best model (lowest validation loss) across all folds
    5. After cross-validation, load the best model and evaluate it on the test set
    6. Construct and return the TrainingOutput object with the required information

  Remember to handle the device (CPU/GPU) correctly throughout the function, especially
  when creating datasets and moving the model between devices.

  Note: This function ties together many components of the training process. Make sure
  to understand how each part (data loading, cross-validation, training, evaluation)
  fits into the overall workflow.

  Parameters
  ----------
  target_variable : TempVarName
    Variable to be predicted.

  in_seq_len : int
    Length of the sequences used as input of the model.
    The shorter it is, the less "history" the RNN will "see" during training.

  out_seq_len : int
    Length of the sequences the model should be trained to forecast.

  hidden_size : int
    Number of neurons in the hidden layer of the model.

  cell_cls : CellClass
    Type of cell in the recurrent part of the model.

  batch_size : int
    Size of the mini-batches used during gradient descent.

  early_stopping_patience : int
    Number of steps until we stop training because the validation loss did not decrease
    (cumulative number of steps).

  valid_loss_eval_freq : int
    Frequency (in number of training batches) at which the validation loss is evaluated.
    Calculating the validation loss at every batch would be creazy, and therefore we
    only calculate it at some frequency.

  device : str
    Device on which the computation should happen ('cpu', 'cuda:0', etc.).

  start_date : date | None
    Optional starting date for the data that should be used.

  end_date : date | None
    Optional ending date for the data that should be used

  Returns
  -------
  TrainingOutput

  """
  boundaries = dlstp.get_daily_paris_temp_dataset_date_boundaries()
  if start_date is None:
    start_date = boundaries[0]
  else:
    start_date = max(start_date, boundaries[0])
  if end_date is None:
    end_date = boundaries[0]
  else:
    end_date = min(end_date, boundaries[1])
  split_date = start_date + timedelta(days=2 * (end_date - start_date).days // 3)
  cross_val_losses: list[MeasuredLoss] = []
  train_dataset = DailyTempSeqDataset(
      start_date=start_date,
      end_date=split_date,
      seq_len=in_seq_len + out_seq_len + 1,
      target_variable=target_variable,
      device=device
  )

  test_dataset = DailyTempSeqDataset(
      start_date=split_date,
      end_date=end_date,
      seq_len=in_seq_len + out_seq_len + 1,
      target_variable=target_variable,
      device=device
  )

  cross_best_model = None
  lowest_lose = 0.0
  fold_id = 0
  cross_losses: list[MeasuredLoss] = []

  for train_subset, valid_subset in iter_cross_val_folds(train_dataset):
    train_loader = DataLoader(train_subset, batch_size=batch_size)
    valid_loader = DataLoader(valid_subset, batch_size=batch_size)

    best_model, losses = train_and_early_stop(
        input_size=train_dataset.data.shape[-1],
        hidden_size=hidden_size,
        cell_cls=cell_cls,
        train_loader=train_loader,
        valid_loader=valid_loader,
        in_seq_len=in_seq_len,
        out_seq_len=out_seq_len,
        early_stopping_patience=early_stopping_patience,
        valid_loss_eval_freq=valid_loss_eval_freq,
        fold_id=fold_id,  # You need to set the correct fold_id
        device=device
    )

    best_loss = min(loss['loss'] for loss in losses if loss['split'] == 'valid')

    validation_loss = [loss for loss in losses if loss["split"] == "valid"]

    cross_val_losses.extend(validation_loss)

    if cross_best_model is None:
      print("cross_best_model is None")
      cross_best_model = best_model
      lowest_lose = best_loss
    else:
      if best_loss < lowest_lose:
        print("best_loss < lowest_lose")
        cross_best_model = best_model
        lowest_lose = best_loss
    
    fold_id += 1

  if (cross_best_model is None):
    raise ValueError("No model was trained during cross validation")
  test_loader = DataLoader(test_dataset, batch_size=batch_size)
  test_loss = evaluate(cross_best_model, test_loader, in_seq_len, out_seq_len, device)
    

  return TrainingOutput(
      best_cross_val_model=cross_best_model,
      cross_val_losses=cross_val_losses,
      test_loss=test_loss
  )



