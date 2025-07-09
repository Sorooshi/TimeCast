# TimeCast

**🌍 Languages:** [English](README.md) | [Русский](README_ru.md) | [فارسی](README_fa.md)

A comprehensive PyTorch-based library for time series forecasting that implements multiple state-of-the-art deep learning models with automated hyperparameter tuning, experiment management, and robust result tracking. **Mathematically validated** against formal LaTeX formulation with complete dimensional correspondence.

## 🚀 Key Features

- **Multiple State-of-the-Art Models**: LSTM, TCN, Transformer, HybridTCNLSTM, MLP
- **Automated Hyperparameter Tuning**: Using Optuna for optimal parameter search
- **Experiment Management**: Organized experiment tracking with custom descriptions
- **3 Training Modes**: Streamlined workflow for different use cases
- **Robust Data Processing**: Clean, efficient preprocessing without artificial time features
- **Professional Data Splitting**: Train/val splits for tune/train, separate test files for predict
- **Automatic Plot Generation**: Training/validation curves saved automatically
- **Complete Training History**: Epoch-by-epoch metrics and progress tracking
- **Merchant Data Preprocessing**: Complete pipeline for transaction-to-timeseries conversion
- **Mathematical Validation**: LaTeX formulation compatibility verified
- **Comprehensive Logging**: Detailed file logging for debugging and analysis
- **Cross-Platform Support**: Robust directory creation across different operating systems
- **Rich Visualization**: Training curves and evaluation plots
- **Modular Architecture**: Clean, maintainable code structure

## 📐 Mathematical Foundation

TimeCast implements the time series forecasting formulation described in our research paper:

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
git clone https://github.com/Sorooshi/TimeCast.git
cd TimeCast
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

| Mode | Description | Data Usage | Artifacts Saved |
|------|-------------|------------|-----------------|
| `tune` | Hyperparameter optimization only | Train/val split from main data | Tuned parameters in hierarchical structure |
| `train` | Training with tuned (`--train_tuned true`) or default (`--train_tuned false`) parameters | Train/val split from main data | **Plots, history, metrics, predictions** in hierarchical directories |
| `predict` | Load trained model and make predictions (`--predict_tuned true/false`) | **Requires separate test data file** | Predictions and metrics in hierarchical structure |
| `report` | Display comprehensive experiment analysis | - | Analysis summaries |

**🎨 File Organization**: All artifacts are now saved in a hierarchical structure:
```
Results/{model}/{mode}/{exp_subdir}/
History/{model}/{mode}/{exp_subdir}/
Predictions/{model}/{mode}/{exp_subdir}/
Metrics/{model}/{mode}/{exp_subdir}/
Hyperparameters/{model}/{mode}/{exp_subdir}/
Plots/{model}/{mode}/{exp_subdir}/
Logs/{model}/{mode}/{exp_subdir}/
Weights/{model}/{mode}/{exp_subdir}/
```
Where `{exp_subdir}` is typically `seq_len_{N}/` or `seq_len_{N}/{experiment_description}/` (and may include `test_{test_data_name}` for predict mode if used).

**🔄 Mode-Specific Directory Creation**:
- `train_tuned/train_default`: Creates results, history, plots, predictions, metrics
- `predict`: Creates only results, predictions, metrics
- `tune`: Creates all directories including hyperparameters

**🎨 New in Train Mode**: Automatically saves training/validation plots (loss, R², MAPE) and complete training history!

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
- `--test_data_name`: **[Predict mode only]** Name of test dataset (without .csv extension)
- `--test_data_path`: **[Predict mode only]** Full path to test data file
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

#### 🔄 Predict Mode Requirements
**Important**: Predict mode now requires separate test data! You must provide either:
- `--test_data_name my_test_data` (uses data/my_test_data.csv)
- `--test_data_path /path/to/test_file.csv` (full path)

This ensures **proper data isolation** and **no data leakage** between training and testing.

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

# Step 5: Make predictions with tuned model (requires separate test data)
python main.py --model Transformer \
               --data_name merchant_processed \
               --test_data_name merchant_test \
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
TimeCast/
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
│   └── {model}/{mode}/{exp_subdir}/
├── 📁 Hyperparameters/         # Tuned and saved parameters
│   └── {model}/{mode}/{exp_subdir}/
├── 📁 Predictions/             # Model predictions
│   └── {model}/{mode}/{exp_subdir}/
├── 📁 Metrics/                 # Detailed evaluation metrics
│   └── {model}/{mode}/{exp_subdir}/
├── 📁 History/                 # Training history (loss curves)
│   └── {model}/{mode}/{exp_subdir}/
├── 📁 Plots/                   # Training visualizations
│   └── {model}/{mode}/{exp_subdir}/
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
Results/Transformer/train/seq_len_5/baseline_experiment/
├── summary.json              # Complete experiment summary
└── plots/
    ├── loss_plot.png         # Training/validation loss curves
    ├── r2_plot.png           # R² score progression
    └── mape_plot.png         # MAPE progression

History/Transformer/train/seq_len_5/baseline_experiment/
└── training_history.csv     # Epoch-by-epoch training data

Predictions/Transformer/train/seq_len_5/baseline_experiment/
├── val_predictions.csv      # Validation predictions vs targets
└── test_predictions.csv     # Test predictions vs targets

Metrics/Transformer/train/seq_len_5/baseline_experiment/
└── metrics.json            # Final evaluation metrics

Hyperparameters/Transformer/train/seq_len_5/baseline_experiment/
├── tune_parameters.json    # Parameters from tuning
└── train_parameters.json   # Parameters used in train mode

Plots/Transformer/train/seq_len_5/baseline_experiment/
├── loss_plot.png           # Automatically generated training plots
├── r2_plot.png             # R² progression visualization  
└── mape_plot.png           # MAPE progression visualization

Logs/Transformer/train/seq_len_5/baseline_experiment/
└── training_log_YYYYMMDD_HHMMSS.txt  # Detailed training logs

Weights/Transformer/train/seq_len_5/baseline_experiment/
└── model_weights.pth        # Saved model weights
```

- For **predict mode**, the structure is similar but may include `test_{test_data_name}` in the path if you use multiple test datasets.
- `{exp_subdir}` is always constructed from sequence length and experiment description (and optionally test dataset name for predict mode).

### 🎨 New Automatic Plot Generation
**Train mode now automatically saves:**
- **Loss curves**: Training vs validation loss progression
- **R² plots**: Model performance over epochs
- **MAPE plots**: Mean Absolute Percentage Error trends
- **Complete history**: CSV file with all epoch metrics

### Key Metrics Tracked
- **Loss**: Mean Squared Error
- **R² Score**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error
- **Training History**: Complete epoch-by-epoch progression

## 🔧 Advanced Features

### 🔄 Professional Data Splitting
**No Data Leakage Guarantee:**
- **Tune/Train modes**: Use train/val splits from same dataset
- **Predict mode**: Requires separate test CSV files
- **Scaler fitting**: Always fitted on training data only
- **Proper isolation**: Test data never seen during training

**Data Flow:**
```
Tune/Train: data.csv → train/val splits → fit scalers on train → normalize both
Predict:    data.csv + test.csv → fit scalers on train → apply to test
```

### 🎨 Automatic Visualization
- **Training plots**: Loss, R², MAPE curves automatically generated
- **Training history**: Complete epoch-by-epoch CSV records
- **Organized storage**: All artifacts saved in structured directories
- **Professional presentation**: Ready-to-use plots for reports/papers

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
3. **Train mode benefits**: Now automatically saves plots and training history!
4. **Predict mode**: Always prepare separate test CSV files for proper evaluation
5. **Experiment descriptions**: Use meaningful names for organization
6. **Data splitting**: Trust the automatic train/val splits - no manual intervention needed
7. **Visualization**: Check the automatically generated plots in Plots/ directory
8. **Training history**: Use the CSV files in History/ for detailed analysis
9. **Logging**: Check log files for detailed training information
10. **Mathematical validation**: Run `test_preprocessing_validation.py` to verify setup

### 🎯 Quick Start Workflow
```bash
# 1. Train with automatic plot generation
python main.py --model LSTM --data_name my_data --mode train --train_tuned false --epochs 50

# 2. Check your results
ls Plots/LSTM/train/seq_len_10/    # View generated plots
ls History/LSTM/train/seq_len_10/  # Check training history

# 3. Make predictions (with separate test file)
python main.py --model LSTM --data_name my_data --test_data_name my_test --mode predict
```

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
  title = {TimeCast: A Comprehensive PyTorch Framework for Time Series Forecasting with LaTeX Formulation Validation},
  author = {Soroosh Shalileh},
  year = {2025},
  url = {https://github.com/Sorooshi/TimeCast},
  note = {TimeCast: Modular time series forecasting with automated hyperparameter tuning and mathematical validation}
}
```

## 📞 Contact

**Author**: Soroosh Shalileh  
**Email**: sr.shalileh@gmail.com  
**GitHub**: [Sorooshi](https://github.com/Sorooshi)

---

## 🆕 Recent Updates (v2.0)

### 🎨 Enhanced Training Experience
- **Automatic Plot Generation**: Train mode now automatically saves loss, R², and MAPE curves
- **Complete Training History**: Epoch-by-epoch metrics saved to CSV files
- **Professional Visualization**: Ready-to-use plots for reports and presentations

### 🔄 Improved Data Handling  
- **Professional Data Splitting**: Clean separation between training and testing data
- **No Data Leakage**: Scalers always fitted on training data only
- **Separate Test Files**: Predict mode requires independent test CSV files
- **Proper Data Flow**: Train/val splits for tune/train, separate test data for predict

### 🛠️ Enhanced CLI
- **New Arguments**: `--test_data_name` and `--test_data_path` for predict mode
- **Better Error Messages**: Clear guidance when files or models are missing
- **Consistent Function Signatures**: All parameter mismatches resolved

### 📊 Complete Artifact Management
- **Organized Structure**: All results saved in structured directories
- **Training Plots**: Automatically generated and saved in Plots/ directory
- **Training History**: Complete CSV records in History/ directory
- **Professional Output**: Everything ready for research and production use

---

*Built with ❤️ for the time series forecasting community*
*Mathematically validated and research-ready ✅*
*Enhanced with professional data handling and automatic visualization 🎨*

### 🔧 Important: Mode Sequence

For using tuned models, follow this sequence:

1. **Tune Mode**: Find best hyperparameters
```bash
python main.py --mode tune --model TCN --data_name my_data
```

2. **Train Mode**: Train with tuned parameters
```bash
python main.py --mode train --model TCN --data_name my_data --train_tuned true
```

3. **Predict Mode**: Use trained model
```bash
python main.py --mode predict --model TCN --data_name my_data --test_data_name test_data --predict_tuned true
```

**Note**: Predict mode requires separate test data file to ensure proper data isolation.
