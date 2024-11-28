# SiamQuality: a ConvNet-based foundation model for photoplethysmography signals

## What is the main problem addressed? In your own words.

The study tackles the challenge of processing and analyzing low-quality PPG signals, which are often compromised 
by noise and artifacts caused by factors like motion or sensor displacement

## How can it help society? What are actual applications of this technology?

This model improves the reliability of health monitoring systems

## What are the limitations of previous approaches?

High sensitivity to noise: Previous models often fail when processing low-quality signals.


## What are the key novelties presented?

Signal Quality Pairing: A method that pairs high-quality and low-quality signals from similar physiological states.
Curriculum Learning: Gradually trains the model with increasingly noisy signals to improve robustness.
Foundation Model Design: Uses CNNs for self-supervised learning, achieving efficiency and robustness over transformers.

## Describe the dataset(s) used? What does it correspond to in the real world?

Domain: Medical (intensive care unit data).
Content: Over 36 million 30-second PPG signal pairs, sampled at 240 Hz and downsampled to 40 Hz.
Real-World Context: Signals were collected from ICU patients, representing real-world noise and artifacts.

## ML training process (the pipeline)

### What data preprocessing and/or curation was used?


Downsampling PPG signals from 240 Hz to 40 Hz.
Min-max normalization.
Signal segmentation into 30-second clips.
Quality assessment to categorize segments as high or low qualit

### Was data augmentation used? If so, describe the process.

No 

### Describe the model's architecture

Base: ResNet backbone (CNN).
Components:
Encoder: Extracts high-level features.
Projector: Maps features into lower-dimensional contrastive space.
Predictor: Aligns paired features using cosine similarity loss.

### Describe the pre-training phase

Task: Contrastive learning using signal quality pairs.
Loss Function: Cosine similarity loss to align representations of paired signals.
Hyperparameters:
Large batch sizes.
Learning rate schedules optimized for convergence.

### Describe how the model was evaluated

Methods: Fine-tuning for downstream tasks (e.g., respiratory rate estimation, atrial fibrillation detection).
Metrics: Mean Absolute Error (MAE) for regression; F1 Score for classification.
Baseline Models: Compared against methods like SimCLR, BYOL, and task-specific models.
Downstream Data: Public datasets for tasks like heart rate estimation and blood pressure monitoring.

### Reproducibility

Code: Publicly available on GitHub.
Pre-trained Weights: Not explicitly mentioned.
Dataset: Pre-training dataset is restricted but downstream datasets are publicly accessible.
