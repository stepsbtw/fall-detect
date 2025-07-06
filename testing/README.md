
# Deep Learning Pipeline for Wearable Sensor Data

This project implements a flexible training pipeline using PyTorch for classifying data from wearable sensors. It supports multiple input types, label schemes, and neural network architectures, including **CNN1D**, **MLP**, and **LSTM**, with **Bayesian hyperparameter optimization via Optuna**.

---

## Project Structure

```
project/
│
├── run.py                      # Main entry point: handles CLI, loading, training, saving
├── train.py                    # Training, evaluation, and Optuna logic
├── neural_networks.py          # CNN1D, MLP, and LSTM architecture definitions
├── labels_and_data/            # Data and label arrays organized by sensor position
│   ├── data/
│   └── labels/
└── output/                     # Automatically created: stores models, metrics, plots
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install torch scikit-learn matplotlib optuna numpy
```

---

### 2. Run Training with Optuna Optimization

```bash
python run.py \
  --scenario Sc1_acc_T \
  --position chest \
  --label_type binary_one \
  --neural_network_type LSTM
```

**Arguments:**

| Argument             | Choices                                                                 | Description                                  |
|----------------------|--------------------------------------------------------------------------|----------------------------------------------|
| `--scenario`         | Sc1_acc_T, Sc1_gyr_T, ..., Sc_4_F                                        | Type of input signal & domain                |
| `--position`         | left, chest, right                                                       | Sensor position                              |
| `--label_type`       | multiple_one, multiple_two, binary_one, binary_two                       | Target label scheme                          |
| `--neural_network_type` | CNN1D, MLP, LSTM                                                       | Type of neural network                       |

---

## Model Architectures

- **CNN1DNet**: 1D convolutional architecture for time/frequency domain signal arrays
- **MLPNet**: Fully connected feedforward network for flat feature vectors
- **LSTMNet**: Recurrent model for sequential data, uses final hidden state for classification

---

## Features

- **Optuna-based hyperparameter optimization**
- Automated model training, evaluation, and saving
- Supports multi-class and binary classification
- Saves:
  - `.pt` model files
  - confusion matrices
  - classification reports
  - ROC curves (for binary)
  - MCC, accuracy, sensitivity, specificity, precision

---

## Output Example

```
output/
└── lstm/
    └── chest/
        └── Sc1_acc_T/
            └── binary_one/
                ├── model_1/
                │   ├── model_1.pt
                │   ├── confusion_matrix_model_1.png
                │   ├── classification_report_model_1.txt
                │   ├── roc_curve_model_1.png
                │   └── metrics_model_1.csv
                └── model_2/
                    ...
```

---

## Notes

- LSTM input shape must be 3D: `(batch_size, sequence_length, input_dim)`
- You can reshape flat arrays if needed: `X.reshape(samples, time_steps, 1)`
- Optimization is repeated for **20 trials**, and final training is repeated **20 times** for robustness