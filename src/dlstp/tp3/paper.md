# <REPLACE WITH PAPER FULL TITLE>

## What is the main problem addressed? In your own words.

TODO

## How can it help society? What are actual applications of this technology?

TODO

## What are the limitations of previous approaches?

TODO

## What are the key novelties presented?

TODO

## Describe the dataset(s) used? What does it correspond to in the real world?

TODO: domain (medical, etc.), multimodal?, sampling rate, sequence length, etc.

## ML training process (the pipeline)

### What data preprocessing and/or curation was used?

TODO: removal of bad data, normalization, segmentation? etc.

### Was data augmentation used? If so, describe the process.

TODO: domain-specific augmentations?

TODO: what invariances are these data augmentations aiming to encode?

### Describe the model's architecture

TODO: CNN, LSTM, Transformer, etc.?

TODO: what is the dimension of the learned representations?

### Describe the pre-training phase

TODO: pretext/upstream tasks, their hypeparameters, contrastive learning?
e.g. positive/negative pairs

TODO: detail the loss that why minimized during pre-training

TODO: what batch sizes were used? what optimiser?

TODO: any specific training tricks? learning rate scheduling, warmup, etc. how did they
avoid dimensional collapse? if applicable

TODO: did they conduct hyperparameter exploration? If so what are the results?

### Describe how the model was evaluated

TODO: methods used (e.g. k-NN zero-shot, linear probing, fine-tuning, etc.)

TODO: detail the downstream tasks, the target variables, the size of the labelled
dataset compared with the size of the unlabelled (pre-training) dataset

TODO: transfer learning? how different is the target domain from the pre-training
domain?

TODO: detail the metrics reported and the results

TODO: against what models are they benchmarking (don't give names of models, instead try
to understand how these baseline models were trained differently compared to the
proposed approach)

TODO: describe if they drop a part of the model (projector head), which part they kept
to obtain the representation used in evaluation

TODO: did they conduct ablation studies? if so, what did they find?

### Reproducibility

TODO: is the code publicly available? what's the licence?

TODO: is the pre-trained model weights publicly available? what's the licence?

TODO: are the dataset and training pipeline publicly available?
