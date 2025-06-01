# Time Series Forecasting Package

A comprehensive PyTorch-based package for time series forecasting that implements multiple state-of-the-art deep learning models.

## Models Implemented

- **LSTM**: Long Short-Term Memory network for sequence modeling
- **TCN**: Temporal Convolutional Network with dilated causal convolutions
- **Transformer**: Self-attention based model for time series
- **HybridTCNLSTM**: A hybrid model combining TCN's hierarchical feature extraction with LSTM's sequential learning
- **MLP**: Multi-Layer Perceptron baseline model
- **PatchTST**: Patch-based Time Series Transformer (Coming Soon)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Sorooshi/Time_Series_Forecasting.git
cd Time_Series_Forecasting
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The package provides a command-line interface for training and evaluating models:

```bash
python main.py --algorithm <MODEL_NAME> \
               --data_name <DATASET_NAME> \
               --data_path <PATH_TO_DATA> \
               --mode <MODE> \
               --n_trials <NUM_TRIALS> \
               --epochs <NUM_EPOCHS>
```

### Arguments

- `--algorithm`: Model to use (LSTM, TCN, Transformer, HybridTCNLSTM, MLP)
- `--data_name`: Name of the dataset
- `--data_path`: Path to the data file (CSV format)
- `--mode`: Operation mode (tune, apply, report)
- `--n_trials`: Number of hyperparameter tuning trials
- `--epochs`: Number of training epochs
- `--sequence_length`: Input sequence length (default: 10)
- `--patience`: Early stopping patience (default: 25)

### Example

```bash
# Hyperparameter tuning
python main.py --algorithm HybridTCNLSTM \
               --data_name merchant_synthetic \
               --data_path data/merchant_synthetic.csv \
               --mode tune \
               --n_trials 100 \
               --epochs 100

# Training with tuned parameters
python main.py --algorithm HybridTCNLSTM \
               --data_name merchant_synthetic \
               --data_path data/merchant_synthetic.csv \
               --mode apply \
               --epochs 100

# View results
python main.py --algorithm HybridTCNLSTM \
               --data_name merchant_synthetic \
               --mode report
```

## Project Structure

```
Time_Series_Forecasting/
├── data/                  # Data directory
├── models/               # Model implementations
│   ├── __init__.py
│   ├── base_model.py
│   ├── lstm.py
│   ├── tcn.py
│   ├── transformer.py
│   ├── hybrid_tcn_lstm.py
│   ├── patch_tst.py
│   └── mlp.py
├── utils/                # Utility functions
│   ├── __init__.py
│   ├── data_preprocessing.py
│   └── training.py
├── Results/              # Training results
├── Hyperparameters/     # Tuned parameters
├── Predictions/         # Model predictions
├── Metrics/             # Evaluation metrics
├── Plots/              # Training plots
├── Logs/               # Training logs
├── main.py             # Main script
└── requirements.txt    # Dependencies
```

## Features

- **Multiple Models**: Implementation of various state-of-the-art architectures
- **Hyperparameter Tuning**: Automated tuning using Optuna
- **Comprehensive Metrics**: R², MAPE, and MSE for evaluation
- **Visualization**: Training curves and prediction plots
- **Early Stopping**: Prevents overfitting
- **Result Management**: Organized storage of results, metrics, and plots

## Data Format

The input data should be a CSV file with:
- A datetime column named 'date' or 'timestamp'
- Feature columns containing numerical values
- The last column is assumed to be the target variable

## Results and Outputs

The package generates comprehensive outputs organized in different directories:
- `Results/`: Summary of training results
- `Hyperparameters/`: Tuned model parameters
- `Predictions/`: Model predictions
- `Metrics/`: Detailed evaluation metrics
- `Plots/`: Training curves and visualizations
- `Logs/`: Training logs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{time_series_forecasting,
  title = {Time Series Forecasting Package},
  author = {Soroosh Shalileh},
  year = {2025},
  url = {https://github.com/Sorooshi/Time_Series_Forecasting}
}
```
