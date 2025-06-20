# Time Series Forecasting Package

A comprehensive PyTorch-based package for time series forecasting that implements multiple state-of-the-art deep learning models with automated hyperparameter tuning, experiment management, and robust result tracking.

## 🚀 Key Features

- **Multiple State-of-the-Art Models**: LSTM, TCN, Transformer, HybridTCNLSTM, MLP, PatchTST
- **Automated Hyperparameter Tuning**: Using Optuna for optimal parameter search
- **Experiment Management**: Organized experiment tracking with custom descriptions
- **4 Training Modes**: Comprehensive workflow for different use cases
- **Robust Data Processing**: Clean, efficient preprocessing without artificial time features
- **Comprehensive Logging**: Detailed file logging for debugging and analysis
- **Cross-Platform Support**: Robust directory creation across different operating systems
- **Rich Visualization**: Training curves and evaluation plots
- **Modular Architecture**: Clean, maintainable code structure

## 📊 Models Implemented

| Model | Description | Use Case |
|-------|-------------|----------|
| **LSTM** | Long Short-Term Memory network | Sequential pattern learning |
| **TCN** | Temporal Convolutional Network | Hierarchical feature extraction |
| **Transformer** | Self-attention based model | Complex temporal dependencies |
| **HybridTCNLSTM** | Combined TCN + LSTM | Best of both architectures |
| **MLP** | Multi-Layer Perceptron | Baseline comparison |
| **PatchTST** | Patch-based Transformer | Efficient transformer variant |

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Sorooshi/Time_Series_Forecasting.git
cd Time_Series_Forecasting
```

2. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Command Line Interface

The package provides a comprehensive CLI with 4 distinct modes:

```bash
python main.py --model <MODEL_NAME> \
               --data_name <DATASET_NAME> \
               --mode <MODE> \
               --experiment_description <DESCRIPTION> \
               [additional options]
```

### 🎯 Training Modes

| Mode | Description | When to Use |
|------|-------------|-------------|
| `tune` | Hyperparameter tuning + training with best params | First time with new data/model |
| `apply` | Training with previously tuned params (or defaults) | Using existing tuned parameters |
| `apply_not_tuned` | Training with default parameters only | Baseline comparison or quick testing |
| `report` | Display saved results from previous runs | Analysis and comparison |

### 📋 Arguments

#### Required Arguments
- `--model`: Model name (LSTM, TCN, Transformer, HybridTCNLSTM, MLP, PatchTST)
- `--data_name`: Dataset name (without .csv extension)

#### Optional Arguments
- `--data_path`: Full path to data file (default: data/{data_name}.csv)
- `--mode`: Training mode (default: apply)
- `--experiment_description`: Custom experiment description (default: seq_len_{sequence_length})
- `--n_trials`: Hyperparameter tuning trials (default: 100)
- `--epochs`: Training epochs (default: 100)
- `--patience`: Early stopping patience (default: 25)
- `--sequence_length`: Input sequence length (default: 10)

### 💡 Example Workflows

#### 1. Complete Workflow: Tune → Apply → Compare

```bash
# Step 1: Hyperparameter tuning
python main.py --model Transformer \
               --data_name merchant_synthetic \
               --mode tune \
               --experiment_description "baseline_experiment" \
               --n_trials 50 \
               --epochs 100

# Step 2: Apply with tuned parameters
python main.py --model Transformer \
               --data_name merchant_synthetic \
               --mode apply \
               --experiment_description "tuned_run" \
               --epochs 100

# Step 3: Compare with default parameters
python main.py --model Transformer \
               --data_name merchant_synthetic \
               --mode apply_not_tuned \
               --experiment_description "default_run" \
               --epochs 100

# Step 4: View all results
python main.py --model Transformer \
               --data_name merchant_synthetic \
               --mode report \
               --experiment_description "baseline_experiment"
```

#### 2. Quick Testing Workflow

```bash
# Quick test with default parameters
python main.py --model LSTM \
               --data_name my_data \
               --mode apply_not_tuned \
               --experiment_description "quick_test" \
               --epochs 20
```

## 🗂️ Project Structure

```
Time_Series_Forecasting/
├── 📁 data/                     # Data files
│   ├── merchant_synthetic.csv
│   └── your_data.csv
├── 📁 models/                   # Model implementations
│   ├── __init__.py
│   ├── base_model.py
│   ├── lstm.py
│   ├── tcn.py
│   ├── transformer.py
│   ├── hybrid_tcn_lstm.py
│   ├── mlp.py
│   └── patch_tst.py
├── 📁 utils/                    # Utility modules
│   ├── __init__.py
│   ├── data_preprocessing.py    # Data loading and preprocessing
│   ├── training.py             # Training and evaluation
│   ├── file_utils.py           # File and directory management
│   ├── visualization.py        # Plotting and visualization
│   ├── config_manager.py       # Hyperparameter management
│   ├── results_manager.py      # Results saving and loading
│   ├── workflow_manager.py     # Training workflow orchestration
│   └── data_utils.py           # Data utilities
├── 📁 Results/                  # Training results and summaries
│   └── {model}/{mode}/{experiment}/
├── 📁 Hyperparameters/         # Tuned and saved parameters
│   └── {model}/{experiment}/
├── 📁 Predictions/             # Model predictions
│   └── {model}/{mode}/{experiment}/
├── 📁 Metrics/                 # Detailed evaluation metrics
│   └── {model}/{mode}/{experiment}/
├── 📁 History/                 # Training history (loss curves)
│   └── {model}/{mode}/{experiment}/
├── 📁 Plots/                   # Training visualizations
│   └── {model}/{mode}/{experiment}/
├── 📁 Logs/                    # Training logs and debugging info
│   └── {model}/
├── main.py                     # Main entry point
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 📊 Data Format

### Input Requirements
- **Format**: CSV file
- **Datetime**: Column named 'date' or 'timestamp' (optional)
- **Features**: Numerical columns representing your time series features
- **No Preprocessing Required**: The system handles normalization automatically

### Example Data Structure
```csv
date,feature1,feature2,feature3,target
2023-01-01,10.5,20.3,5.7,100.2
2023-01-02,11.2,19.8,6.1,102.5
...
```

## 📈 Outputs and Results

### Organized Experiment Structure
Each experiment creates a complete directory structure:

```
Results/Transformer/tune/baseline_experiment/
├── summary.json              # Complete experiment summary
└── plots/
    ├── loss_plot.png         # Training/validation loss
    ├── r2_plot.png           # R² score progression
    └── mape_plot.png         # MAPE progression

History/Transformer/tune/baseline_experiment/
└── training_history.csv     # Epoch-by-epoch training data

Predictions/Transformer/tune/baseline_experiment/
├── val_predictions.csv      # Validation predictions vs targets
└── test_predictions.csv     # Test predictions vs targets

Metrics/Transformer/tune/baseline_experiment/
└── metrics.json            # Final evaluation metrics

Hyperparameters/Transformer/baseline_experiment/
├── tune_parameters.json    # Parameters from tuning
└── apply_parameters.json   # Parameters used in apply mode

Logs/Transformer/
└── tuning_log_20250620_HHMMSS.txt  # Detailed training logs
```

### Key Metrics Tracked
- **Loss**: Mean Squared Error
- **R² Score**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error
- **Training History**: Complete epoch-by-epoch progression

## 🔧 Advanced Features

### Experiment Management
- **Custom Descriptions**: Organize experiments with meaningful names
- **Automatic Fallback**: Uses sequence length if no description provided
- **Safe Naming**: Automatically handles special characters in experiment names

### Robust Architecture
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Error Handling**: Comprehensive error checking and reporting
- **Modular Design**: Easy to extend and modify
- **Clean Data Processing**: No artificial time features for better compatibility

### Logging and Debugging
- **File Logging**: Detailed logs saved for each tuning session
- **Trial Tracking**: Individual hyperparameter trial results
- **Progress Monitoring**: Real-time training progress
- **Error Tracking**: Comprehensive error logging

## 🚀 Performance Tips

1. **Start with tuning**: Use `--mode tune` for new datasets
2. **Use apply_not_tuned**: For quick baselines and comparisons  
3. **Experiment descriptions**: Use meaningful names for organization
4. **Logging**: Check log files for detailed training information
5. **Cross-validation**: Results are automatically validated on separate test sets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this package in your research, please cite:

```bibtex
@software{time_series_forecasting_2025,
  title = {Time Series Forecasting Package: A Comprehensive PyTorch Framework},
  author = {Soroosh Shalileh},
  year = {2025},
  url = {https://github.com/Sorooshi/Time_Series_Forecasting},
  note = {Modular time series forecasting with automated hyperparameter tuning}
}
```

## 📞 Contact

**Author**: Soroosh Shalileh  
**Email**: sr.shalileh@gmail.com  
**GitHub**: [Sorooshi](https://github.com/Sorooshi)

---

*Built with ❤️ for the time series forecasting community*
