# Time Series Forecasting Package

**🌍 Languages:** [English](README.md) | [Русский](README_ru.md) | [فارسی](README_fa.md)

A comprehensive PyTorch-based package for time series forecasting that implements multiple state-of-the-art deep learning models with automated hyperparameter tuning, experiment management, and robust result tracking. **Mathematically validated** against formal LaTeX formulation with complete dimensional correspondence.

## 🚀 Key Features

- **Multiple State-of-the-Art Models**: LSTM, TCN, Transformer, HybridTCNLSTM, MLP
- **Automated Hyperparameter Tuning**: Using Optuna for optimal parameter search
- **Experiment Management**: Organized experiment tracking with custom descriptions
- **3 Training Modes**: Streamlined workflow for different use cases
- **Robust Data Processing**: Clean, efficient preprocessing without artificial time features
- **Merchant Data Preprocessing**: Complete pipeline for transaction-to-timeseries conversion
- **Mathematical Validation**: LaTeX formulation compatibility verified
- **Comprehensive Logging**: Detailed file logging for debugging and analysis
- **Cross-Platform Support**: Robust directory creation across different operating systems
- **Rich Visualization**: Training curves and evaluation plots
- **Modular Architecture**: Clean, maintainable code structure

## 📐 Mathematical Foundation

This package implements the time series forecasting formulation described in our research paper:

### Problem Formulation
Given merchant-level transaction data, we forecast total consumption using historical sequences:

**LaTeX Notation → Implementation Mapping:**
- Historical sequence: $\mathcal{H}_t \in \mathbb{R}^{(k+1) \times N}$ ↔ `(sequence_length, n_features)`
- Merchant consumption: $X_t \in \mathbb{R}^N$ ↔ `merchant_features[t]`
- Target prediction: $y_t = \sum_{m=1}^N x_{m,t}$ ↔ `np.sum(data[t])`

**✅ Dimensional Compatibility Verified:**
```
LaTeX: 𝒽_t ∈ ℝ^{(k+1)×N}  ↔  Implementation: (batch_size, sequence_length, n_features)
```

## 📊 Models Implemented

| Model | Description | Use Case | Paper Reference |
|-------|-------------|----------|------------------|
| **LSTM** | Long Short-Term Memory network | Sequential pattern learning | Hochreiter & Schmidhuber (1997) |
| **TCN** | Temporal Convolutional Network | Hierarchical feature extraction | Bai et al. (2018) |
| **Transformer** | Self-attention based model | Complex temporal dependencies | Vaswani et al. (2017) |
| **HybridTCNLSTM** | Combined TCN + LSTM | Best of both architectures | Custom Implementation |
| **MLP** | Multi-Layer Perceptron | Baseline comparison | Zhang et al. (1998) |

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

### Quick Start with Merchant Data

For merchant transaction data preprocessing (recommended starting point):

```bash
# Step 1: Run the preprocessing example
python example.py

# Step 2: Train models on preprocessed data with all arguments
python main.py --model Transformer \
               --data_name merchant_processed \
               --data_path data/merchant_processed.csv \
               --mode train \
               --train_tuned false \
               --experiment_description "merchant_baseline" \
               --n_trials 100 \
               --epochs 100 \
               --patience 25 \
               --sequence_length 5
```

### Command Line Interface

The package provides a comprehensive CLI with 3 distinct modes:

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
| `tune` | Hyperparameter optimization only | First time with new data/model |
| `train` | Training with tuned (`--train_tuned true`) or default (`--train_tuned false`) parameters | Main training mode |
| `predict` | Load trained model and make predictions (`--predict_tuned true/false`) | Making predictions |
| `report` | Display comprehensive experiment analysis | Analysis and comparison |

### 📊 Report Mode Features

The report mode provides comprehensive analysis of your experiments with multiple view options:

```bash
python main.py --model <MODEL> --data_name <DATA> --mode report --report_type <TYPE>
```

#### Report Types Available:

| Report Type | Description | Shows |
|-------------|-------------|--------|
| `all` | Complete comprehensive report | Everything combined |
| `models` | Available trained models | Model status, completeness |
| `performance` | Performance comparison tables | Best metrics, rankings |
| `best` | Best performing configurations | Top 5 configurations with hyperparameters |
| `timeline` | Experiment timeline | Chronological experiment history |
| `files` | File paths and storage info | Directory structure, file sizes |

#### 🔍 Report Examples:

**Show all available models:**
```bash
python main.py --model LSTM --data_name test_data --mode report --report_type models
```

**Performance comparison:**
```bash
python main.py --model LSTM --data_name test_data --mode report --report_type performance
```

**Complete analysis:**
```bash
python main.py --model LSTM --data_name test_data --mode report --report_type all
```

#### 📈 What Each Report Shows:

**🤖 Models Report:**
- Available trained models with status (Complete/Partial/No Weights)
- Tuned vs default model availability
- Model completeness statistics
- Experiment organization overview

**📊 Performance Report:**
- Best performance by model type
- Detailed performance rankings
- Test loss, R², and MAPE comparisons
- Performance trends across experiments

**🏆 Best Configurations Report:**
- Top 5 best performing configurations
- Key hyperparameters for best models
- Performance metrics for each configuration
- Hyperparameter recommendations

**⏰ Timeline Report:**
- Chronological experiment history
- File modification timestamps
- Experiment frequency analysis
- Date range summaries

**📁 Files Report:**
- Complete directory structure
- Hyperparameter files with sizes and dates
- Weight files with storage information
- Results files organization
- Total storage usage statistics

### 📋 Arguments

#### Required Arguments
- `--model`: Model name (LSTM, TCN, Transformer, HybridTCNLSTM, MLP)
- `--data_name`: Dataset name (without .csv extension)

#### Optional Arguments
- `--data_path`: Full path to data file (default: data/{data_name}.csv)
- `--mode`: Training mode (default: train)
- `--experiment_description`: Custom experiment description (default: seq_len_{sequence_length})
- `--train_tuned`: Whether to use tuned parameters for training (true/false, default: true)
- `--predict_tuned`: Whether to use tuned model for prediction (true/false, default: true)
- `--report_type`: Type of report to show (all/models/performance/best/timeline/files, default: all)
- `--n_trials`: Hyperparameter tuning trials (default: 100)
- `--epochs`: Training epochs (default: 100)
- `--patience`: Early stopping patience (default: 25)
- `--sequence_length`: Input sequence length (default: 10)
- `--k_folds`: Number of folds for K-fold cross validation (default: 5)

### 🔧 Important: Data Path Usage

**Common Mistake:** Don't point `--data_path` to a directory!

```bash
# ❌ WRONG - This will fail
python main.py --model LSTM --data_name my_data --data_path data/

# ✅ CORRECT - Specify the complete file path
python main.py --model LSTM --data_name my_data --data_path data/my_data.csv

# ✅ RECOMMENDED - Let the system auto-construct the path
python main.py --model LSTM --data_name my_data
# This automatically uses: data/my_data.csv
```

**Key Points:**
- `--data_path` expects a **file path**, not a directory
- If omitted, the system constructs: `data/{data_name}.csv`
- Always include the `.csv` extension when specifying `--data_path`

## 🏪 Merchant Data Preprocessing

### Preprocessing Pipeline (`example.py`)

Complete pipeline for converting raw merchant transaction data to time series format:

```bash
python example.py
```

**Pipeline Steps:**
1. **Load Transaction Data**: Raw transaction-level data loading
2. **Merchant Aggregation**: Group by time periods and merchants
3. **Contextual Features**: Add time-based features (seasonality, holidays, etc.)
4. **LaTeX Compatibility**: Ensure dimensional correspondence
5. **Validation**: Test with TimeSeriesPreprocessor

**Input Format:**
```csv
timestamp,merchant_id,customer_id,amount,day_of_week,hour,is_weekend,is_holiday,transaction_speed,customer_loyalty_score
2023-01-01 03:41:00,1,23,16.02,6,3,True,False,8.87,79.8
2023-01-01 06:28:00,4,25,99.56,6,6,True,False,5.9,48.8
...
```

**Output Format:**
```csv
date,merchant_1,merchant_2,merchant_3,merchant_4,merchant_5,hour,day_of_week,is_weekend,month,day_of_month,sin_month,cos_month,sin_hour,cos_hour,is_holiday
2023-01-01,454.17,207.98,216.56,460.11,644.78,0,5,1.0,1,1,0.0,1.0,0.0,1.0,1.0
2023-01-02,423.89,189.45,234.12,501.23,678.91,0,0,0.0,1,2,0.0,1.0,0.0,1.0,0.0
...
```

### 💡 Example Workflows

#### 1. Complete Merchant Data Workflow

```bash
# Step 1: Preprocess merchant data
python example.py

# Step 2: Hyperparameter tuning
python main.py --model Transformer \
               --data_name merchant_processed \
               --mode tune \
               --experiment_description "merchant_baseline" \
               --n_trials 50 \
               --epochs 100 \
               --sequence_length 5

# Step 3: Train with tuned parameters (K-fold CV)
python main.py --model Transformer \
               --data_name merchant_processed \
               --mode train \
               --train_tuned true \
               --experiment_description "merchant_tuned" \
               --epochs 100 \
               --sequence_length 5

# Step 4: Compare with default parameters
python main.py --model Transformer \
               --data_name merchant_processed \
               --mode train \
               --train_tuned false \
               --experiment_description "merchant_default" \
               --epochs 100 \
               --sequence_length 5

# Step 5: Make predictions with tuned model
python main.py --model Transformer \
               --data_name merchant_processed \
               --mode predict \
               --predict_tuned true \
               --experiment_description "merchant_tuned" \
               --sequence_length 5

# Step 6: View all results
python main.py --model Transformer \
               --data_name merchant_processed \
               --mode report \
               --experiment_description "merchant_baseline"
```

#### 2. Quick Testing Workflow

```bash
# Quick test with default parameters
python main.py --model LSTM \
               --data_name my_data \
               --mode train \
               --train_tuned false \
               --experiment_description "quick_test" \
               --epochs 20 \
               --sequence_length 5
```

## 🧪 Testing and Validation

### Mathematical Validation

Verify LaTeX formulation compatibility:

```bash
python test_preprocessing_validation.py
```

**Validates:**
- ✅ Dimensional correspondence: $(k+1) \times N$ ↔ `(sequence_length, n_features)`
- ✅ Target calculation: $y_t = \sum_{m=1}^N x_{m,t}$ ↔ `np.sum(...)`
- ✅ Preprocessing pipeline compatibility
- ✅ Integration with existing models

### Comprehensive Testing

Run full test suite:

```bash
# Move to test directory
cd Test

# Run comprehensive tests
python test_script.py

# Test feature dimensions  
python test_feature_dimensions.py

# Validate preprocessing
python test_preprocessing_validation.py
```

## 🗂️ Project Structure

```
Time_Series_Forecasting/
├── 📁 data/                     # Data files
│   ├── merchant_synthetic.csv  # Sample merchant data
│   ├── merchant_processed.csv  # Preprocessed merchant data
│   └── your_data.csv
├── 📁 models/                   # Model implementations
│   ├── __init__.py
│   ├── base_model.py
│   ├── lstm.py
│   ├── tcn.py
│   ├── transformer.py
│   ├── hybrid_tcn_lstm.py
│   └── mlp.py
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
├── 📁 Test/                     # Testing and validation
│   ├── test_script.py          # Comprehensive test suite
│   ├── test_feature_dimensions.py  # Feature dimension testing
│   └── test_preprocessing_validation.py  # LaTeX compatibility validation
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
├── example.py                  # Merchant data preprocessing pipeline
├── main.py                     # Main entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── README_ru.md               # Russian version
└── a01_TS_forecasting.tex     # LaTeX research paper
```

## 📊 Data Format

### Input Requirements
- **Format**: CSV file
- **Datetime**: Column named 'date' or 'timestamp' (optional)
- **Features**: Numerical columns representing your time series features
- **No Preprocessing Required**: The system handles normalization automatically

### Raw Transaction Data Structure (for `example.py`)
```csv
timestamp,merchant_id,customer_id,amount,day_of_week,hour,is_weekend,is_holiday,transaction_speed,customer_loyalty_score
2023-01-01 03:41:00,1,23,16.02,6,3,True,False,8.87,79.8
2023-01-01 06:28:00,4,25,99.56,6,6,True,False,5.9,48.8
...
```

### Processed Time Series Data Structure
```csv
date,merchant_1,merchant_2,merchant_3,merchant_4,merchant_5,hour,day_of_week,is_weekend,month,day_of_month,sin_month,cos_month,sin_hour,cos_hour,is_holiday
2023-01-01,454.17,207.98,216.56,460.11,644.78,0,5,1.0,1,1,0.0,1.0,0.0,1.0,1.0
2023-01-02,423.89,189.45,234.12,501.23,678.91,0,0,0.0,1,2,0.0,1.0,0.0,1.0,0.0
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

### Mathematical Foundation
- **LaTeX Formulation**: Implements formal mathematical framework
- **Dimensional Validation**: Automatic dimensional correspondence checking
- **Target Consistency**: Validated target calculation: $y_t = \sum_{m=1}^N x_{m,t}$

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

1. **Start with example.py**: For merchant data, use the preprocessing pipeline
2. **Use tuning mode**: Use `--mode tune` for new datasets
3. **Use train with --train_tuned false**: For quick baselines and comparisons  
4. **Experiment descriptions**: Use meaningful names for organization
5. **Logging**: Check log files for detailed training information
6. **Cross-validation**: Results are automatically validated on separate test sets
7. **Mathematical validation**: Run `test_preprocessing_validation.py` to verify setup

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
  title = {Time Series Forecasting Package: A Comprehensive PyTorch Framework with LaTeX Formulation Validation},
  author = {Soroosh Shalileh},
  year = {2025},
  url = {https://github.com/Sorooshi/Time_Series_Forecasting},
  note = {Modular time series forecasting with automated hyperparameter tuning and mathematical validation}
}
```

## 📞 Contact

**Author**: Soroosh Shalileh  
**Email**: sr.shalileh@gmail.com  
**GitHub**: [Sorooshi](https://github.com/Sorooshi)

---

*Built with ❤️ for the time series forecasting community*
*Mathematically validated and research-ready ✅*
