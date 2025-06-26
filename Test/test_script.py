#!/usr/bin/env python3
"""
Comprehensive Test Script for Time Series Forecasting Package
Tests all models, modes, and functionality with the new four-mode system.

Author: Soroosh Shalileh
Email: sr.shalileh@gmail.com
Year: 2025
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json
from datetime import datetime, timedelta

# Add the root directory to the path so we can import our modules
script_dir = Path(__file__).parent
root_dir = script_dir.parent
sys.path.insert(0, str(root_dir))

# Change working directory to root if we're in Test directory
if os.getcwd().endswith('Test'):
    os.chdir(root_dir)

class TimeSeriesTester:
    """Comprehensive tester for the time series forecasting package."""
    
    def __init__(self):
        self.test_results = []
        self.failed_tests = []
        # Ensure we're working from the project root
        if os.getcwd().endswith('Test'):
            os.chdir('..')
        self.base_dir = Path(".")
        self.temp_data_path = None
        
    def log_test(self, test_name: str, status: str, message: str = ""):
        """Log test results."""
        result = {
            'test': test_name,
            'status': status,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_emoji} {test_name}: {status}")
        if message:
            print(f"   └── {message}")
    
    def create_test_data(self) -> Path:
        """Create synthetic test data."""
        print("\n📊 Creating test data...")
        
        # Create date range
        dates = pd.date_range('2023-01-01', periods=200, freq='H')
        
        # Create synthetic features
        np.random.seed(42)
        n_features = 5
        data = np.random.randn(200, n_features).cumsum(axis=0)
        
        # Add some trend and seasonality
        trend = np.linspace(0, 10, 200).reshape(-1, 1)
        seasonal = 2 * np.sin(2 * np.pi * np.arange(200) / 24).reshape(-1, 1)
        
        data[:, 0] += trend.flatten() + seasonal.flatten()
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)])
        df['date'] = dates
        
        # Reorder columns to have date first
        df = df[['date'] + [f'feature_{i}' for i in range(n_features)]]
        
        # Save to data directory
        data_dir = self.base_dir / "data"
        data_dir.mkdir(exist_ok=True)
        temp_file = data_dir / "test_data.csv"
        df.to_csv(temp_file, index=False)
        self.temp_data_path = temp_file
        
        self.log_test("Create Test Data", "PASS", f"Created {len(df)} rows with {n_features} features")
        return temp_file
    
    def test_models_available(self):
        """Test that all models can be imported."""
        print("\n🤖 Testing model availability...")
        
        models_to_test = ['LSTM', 'TCN', 'Transformer', 'HybridTCNLSTM', 'MLP']
        
        for model_name in models_to_test:
            try:
                # Try to import the model
                import importlib
                models = importlib.import_module('models')
                model_class = getattr(models, model_name)
                
                # Test that we can get default parameters
                default_params = model_class.get_default_parameters()
                assert isinstance(default_params, dict)
                
                self.log_test(f"Model {model_name}", "PASS", f"Successfully imported and has {len(default_params)} default parameters")
                
            except Exception as e:
                self.log_test(f"Model {model_name}", "FAIL", str(e))
                self.failed_tests.append(f"Model {model_name}")
    
    def test_mode(self, model: str, mode: str, epochs: int = 2, n_trials: int = 2, k_folds: int = 2, train_tuned: bool = True, predict_tuned: bool = True):
        """Test a specific model and mode combination with the new three-mode system."""
        mode_desc = mode
        if mode == 'train':
            mode_desc = f"{mode} ({'tuned' if train_tuned else 'default'})"
        elif mode == 'predict':
            mode_desc = f"{mode} ({'tuned' if predict_tuned else 'default'})"
            
        print(f"\n🧪 Testing {model} in {mode_desc} mode...")
        
        try:
            # Build base command
            if mode == 'train':
                experiment_name = f"test_{mode}_{('tuned' if train_tuned else 'default')}_{model.lower()}"
            elif mode == 'predict':
                experiment_name = f"test_{mode}_{('tuned' if predict_tuned else 'default')}_{model.lower()}"
            else:
                experiment_name = f"test_{mode}_{model.lower()}"
            cmd = [
                sys.executable, "main.py",
                "--model", model,
                "--data_name", "test_data",
                "--mode", mode,
                "--epochs", str(epochs),
                "--patience", "2",
                "--experiment_description", experiment_name,
                "--sequence_length", "5"
            ]
            
            # Add mode-specific parameters
            if mode == 'tune':
                cmd.extend(["--n_trials", str(n_trials)])
            elif mode == 'train':
                cmd.extend(["--k_folds", str(k_folds)])
                cmd.extend(["--train_tuned", "true" if train_tuned else "false"])
            elif mode == 'predict':
                cmd.extend(["--predict_tuned", "true" if predict_tuned else "false"])
            
            # Run command
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                # Check if expected files were created based on mode
                expected_files = self._get_expected_files(model, mode, experiment_name)
                missing_files = []
                
                for file_type, file_path in expected_files.items():
                    if not file_path.exists():
                        missing_files.append(file_type)
                
                if missing_files:
                    self.log_test(f"{model} {mode_desc}", "WARN", f"Missing files: {missing_files}")
                else:
                    self.log_test(f"{model} {mode_desc}", "PASS", "All expected files created")
                    
            else:
                error_msg = result.stderr.strip() if result.stderr else result.stdout.strip()
                if not error_msg:
                    error_msg = f"Command failed with return code {result.returncode}"
                self.log_test(f"{model} {mode_desc}", "FAIL", error_msg[:150])
                self.failed_tests.append(f"{model} {mode_desc}")
                
        except subprocess.TimeoutExpired:
            self.log_test(f"{model} {mode_desc}", "FAIL", "Test timed out after 2 minutes")
            self.failed_tests.append(f"{model} {mode_desc}")
        except Exception as e:
            self.log_test(f"{model} {mode_desc}", "FAIL", str(e))
            self.failed_tests.append(f"{model} {mode_desc}")
    
    def test_predict_with_existing_model(self, model: str, predict_tuned: bool):
        """Test predict mode using existing trained model."""
        mode_desc = f"predict ({'tuned' if predict_tuned else 'default'})"
        print(f"\n🧪 Testing {model} in {mode_desc} mode...")
        
        try:
            # Use the experiment name that corresponds to the training that created the weights
            if predict_tuned:
                experiment_name = f"test_train_tuned_{model.lower()}"
            else:
                experiment_name = f"test_train_default_{model.lower()}"
            
            cmd = [
                sys.executable, "main.py",
                "--model", model,
                "--data_name", "test_data",
                "--mode", "predict",
                "--experiment_description", experiment_name,
                "--sequence_length", "5",
                "--predict_tuned", "true" if predict_tuned else "false"
            ]
            
            # Run command
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                # Check if expected files were created
                expected_files = self._get_expected_files(model, "predict", experiment_name)
                missing_files = []
                
                for file_type, file_path in expected_files.items():
                    if not file_path.exists():
                        missing_files.append(file_type)
                
                if missing_files:
                    self.log_test(f"{model} {mode_desc}", "WARN", f"Missing files: {missing_files}")
                else:
                    self.log_test(f"{model} {mode_desc}", "PASS", "All expected files created")
                    
            else:
                error_msg = result.stderr.strip() if result.stderr else result.stdout.strip()
                if not error_msg:
                    error_msg = f"Command failed with return code {result.returncode}"
                self.log_test(f"{model} {mode_desc}", "FAIL", error_msg[:150])
                self.failed_tests.append(f"{model} {mode_desc}")
                
        except subprocess.TimeoutExpired:
            self.log_test(f"{model} {mode_desc}", "FAIL", "Test timed out after 2 minutes")
            self.failed_tests.append(f"{model} {mode_desc}")
        except Exception as e:
            self.log_test(f"{model} {mode_desc}", "FAIL", str(e))
            self.failed_tests.append(f"{model} {mode_desc}")
    
    def _get_expected_files(self, model: str, mode: str, experiment: str):
        """Get expected files for the new four-mode system."""
        unique_specifier = f"{model}_test_data_{experiment}_5"  # sequence_length = 5
        expected_files = {}
        
        if mode == 'tune':
            expected_files['tuned_hyperparams'] = self.base_dir / "Hyperparameters" / f"{unique_specifier}_tuned.json"
            expected_files['logs'] = self.base_dir / "Logs" / model / f"tune_log_*.txt"
            
        elif mode == 'train':
            if 'tuned' in experiment:
                expected_files['tuned_weights'] = self.base_dir / "Weights" / f"{unique_specifier}_tuned_best.pth"
            else:  # default training
                expected_files['default_weights'] = self.base_dir / "Weights" / f"{unique_specifier}_default_best.pth"
                expected_files['default_hyperparams'] = self.base_dir / "Hyperparameters" / f"{unique_specifier}_train.json"
            expected_files['logs'] = self.base_dir / "Logs" / model / f"train_log_*.txt"
            
        elif mode == 'predict':
            expected_files['predictions'] = self.base_dir / "Predictions" / model / mode / experiment
            expected_files['metrics'] = self.base_dir / "Metrics" / model / mode / experiment / "metrics.json"
            expected_files['logs'] = self.base_dir / "Logs" / model / f"predict_log_*.txt"
        
        # For log files with timestamps, just check if the directory exists
        for key, path in expected_files.items():
            if 'logs' in key and '*' in str(path):
                log_dir = path.parent
                if log_dir.exists() and any(log_dir.glob(path.name)):
                    expected_files[key] = log_dir  # Change to directory check
                    
        return expected_files
    
    def test_full_workflow(self, model: str = "LSTM"):
        """Test the complete four-mode workflow."""
        print(f"\n🔄 Testing full workflow for {model}...")
        
        experiment_name = f"full_workflow_{model.lower()}"
        
        # Step 1: Tune hyperparameters
        self.test_mode(model, "tune", epochs=2, n_trials=2)
        if f"{model} tune" in self.failed_tests:
            self.log_test(f"Full Workflow {model}", "FAIL", "Tune step failed")
            return
        
        # Step 2: Train with tuned parameters
        self.test_mode(model, "train", epochs=2, k_folds=2)
        if f"{model} train" in self.failed_tests:
            self.log_test(f"Full Workflow {model}", "FAIL", "Train step failed")
            return
        
        # Step 3: Train with default parameters (independent)
        self.test_mode(model, "train", epochs=2, train_tuned=False)
        if f"{model} train (default)" in self.failed_tests:
            self.log_test(f"Full Workflow {model}", "FAIL", "Train default step failed")
            return
        
        # Step 4: Predict with tuned model
        cmd_tuned = [
            sys.executable, "main.py",
            "--model", model,
            "--data_name", "test_data",
            "--mode", "predict",
            "--experiment_description", f"test_train_tuned_{model.lower()}",
            "--sequence_length", "5",
            "--predict_tuned", "true"
        ]
        
        result_tuned = subprocess.run(cmd_tuned, capture_output=True, text=True, timeout=60)
        
        # Step 5: Predict with default model
        cmd_default = [
            sys.executable, "main.py",
            "--model", model,
            "--data_name", "test_data",
            "--mode", "predict",
            "--experiment_description", f"test_train_default_{model.lower()}",
            "--sequence_length", "5",
            "--predict_tuned", "false"
        ]
        
        result_default = subprocess.run(cmd_default, capture_output=True, text=True, timeout=60)
        
        if result_tuned.returncode == 0 and result_default.returncode == 0:
            self.log_test(f"Full Workflow {model}", "PASS", "All four modes completed successfully")
        else:
            error_msg = "Prediction steps failed"
            if result_tuned.returncode != 0:
                error_msg += f" (tuned: {result_tuned.stderr[:50]})"
            if result_default.returncode != 0:
                error_msg += f" (default: {result_default.stderr[:50]})"
            self.log_test(f"Full Workflow {model}", "FAIL", error_msg)
    
    def test_unique_specifier_system(self):
        """Test that the unique specifier system works correctly."""
        print("\n🏷️ Testing unique specifier system...")
        
        try:
            from utils.file_utils import create_unique_specifier
            
            # Test different combinations
            test_cases = [
                ("LSTM", "test_data", 10, "my_exp", "LSTM_test_data_my_exp_10"),
                ("TCN", "merchant_data", 20, None, "TCN_merchant_data_No_Description_20"),
                ("MLP", "air_quality", 5, "experiment_1", "MLP_air_quality_experiment_1_5")
            ]
            
            for model, data, seq_len, exp_desc, expected in test_cases:
                result = create_unique_specifier(model, data, seq_len, exp_desc)
                if result == expected:
                    self.log_test(f"Unique Specifier {model}", "PASS", f"Generated: {result}")
                else:
                    self.log_test(f"Unique Specifier {model}", "FAIL", f"Expected: {expected}, Got: {result}")
                    
        except Exception as e:
            self.log_test("Unique Specifier System", "FAIL", str(e))
    
    def test_report_mode(self):
        """Test report mode functionality."""
        print("\n📊 Testing report mode...")
        
        try:
            cmd = [
                sys.executable, "main.py",
                "--model", "LSTM",
                "--data_name", "test_data",
                "--mode", "report"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Check if report contains expected information
                output = result.stdout
                if "Report Mode" in output:
                    self.log_test("Report Mode", "PASS", "Report generated successfully")
                else:
                    self.log_test("Report Mode", "WARN", "Report generated but content may be incomplete")
            else:
                self.log_test("Report Mode", "FAIL", result.stderr.strip() if result.stderr else "Unknown error")
                
        except Exception as e:
            self.log_test("Report Mode", "FAIL", str(e))
    
    def test_help_message(self):
        """Test that help message works."""
        print("\n❓ Testing help message...")
        
        try:
            cmd = [sys.executable, "main.py", "--help"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                help_text = result.stdout
                # Check for key elements in help message
                required_elements = ['--model', '--data_name', '--mode', 'tune', 'train', 'predict', '--train_tuned', '--predict_tuned']
                missing_elements = [elem for elem in required_elements if elem not in help_text]
                
                if not missing_elements:
                    self.log_test("Help Message", "PASS", "All required elements present")
                else:
                    self.log_test("Help Message", "WARN", f"Missing elements: {missing_elements}")
            else:
                self.log_test("Help Message", "FAIL", "Help command failed")
                
        except Exception as e:
            self.log_test("Help Message", "FAIL", str(e))
    
    def test_data_processing(self):
        """Test data processing functionality."""
        print("\n📈 Testing data processing...")
        
        try:
            from utils.data_utils import get_data_path, load_and_validate_data
            from utils.data_preprocessing import prepare_data_for_model
            
            # Test data loading
            data_path = get_data_path("test_data", None)
            data, dates = load_and_validate_data(data_path)
            
            if data is not None and len(data) > 0:
                self.log_test("Data Loading", "PASS", f"Loaded {len(data)} rows")
                
                # Test data preparation
                train_loader, val_loader, test_loader, input_size = prepare_data_for_model(
                    data=data,
                    dates=dates,
                    sequence_length=5,
                    train_ratio=0.7,
                    val_ratio=0.15
                )
                
                if train_loader and val_loader and test_loader and input_size > 0:
                    self.log_test("Data Processing", "PASS", f"Input size: {input_size}")
                else:
                    self.log_test("Data Processing", "FAIL", "Failed to create data loaders")
            else:
                self.log_test("Data Loading", "FAIL", "Failed to load data")
                
        except Exception as e:
            self.log_test("Data Processing", "FAIL", str(e))
    
    def test_directory_creation(self):
        """Test directory creation functionality."""
        print("\n📁 Testing directory creation...")
        
        try:
            from utils.file_utils import create_directory_safely, create_experiment_directories
            
            # Test safe directory creation
            test_dir = self.base_dir / "test_temp_dir"
            if create_directory_safely(test_dir):
                self.log_test("Directory Creation", "PASS", "Successfully created test directory")
                # Clean up
                if test_dir.exists():
                    test_dir.rmdir()
            else:
                self.log_test("Directory Creation", "FAIL", "Failed to create test directory")
                
            # Test experiment directory creation
            dirs = create_experiment_directories("TEST_MODEL", "test_mode", "test_exp", 10)
            
            created_dirs = [name for name, path in dirs.items() if path.exists()]
            if len(created_dirs) > 0:
                self.log_test("Experiment Directories", "PASS", f"Created {len(created_dirs)} directories")
                # Clean up
                self._cleanup_test_directories(dirs)
            else:
                self.log_test("Experiment Directories", "FAIL", "No directories were created")
                
        except Exception as e:
            self.log_test("Directory Management", "FAIL", str(e))
    
    def _cleanup_test_directories(self, dirs):
        """Clean up test directories."""
        for dir_path in dirs.values():
            try:
                if dir_path.exists():
                    if dir_path.is_dir():
                        shutil.rmtree(dir_path)
                    else:
                        dir_path.unlink()
            except Exception:
                pass  # Ignore cleanup errors
    
    def run_comprehensive_test(self):
        """Run comprehensive tests for the new four-mode system."""
        print("🚀 Starting Comprehensive Test Suite for Four-Mode System")
        print("=" * 80)
        
        start_time = datetime.now()
        
        # Create test data first
        self.create_test_data()
        
        # Basic functionality tests
        self.test_models_available()
        self.test_help_message()
        self.test_data_processing()
        self.test_directory_creation()
        self.test_unique_specifier_system()
        
        # Test individual modes with quick settings
        test_models = ["LSTM", "MLP"]  # Test with two different models
        test_modes = ["tune", "train", "predict"]  # Test all three modes
        
        for model in test_models:
            for mode in test_modes:
                if mode == "predict":
                    # For predict mode, we need to have trained models first
                    # Train with tuned parameters 
                    self.test_mode(model, "train", epochs=1, n_trials=2, k_folds=2, train_tuned=True)
                    # Train with default parameters
                    self.test_mode(model, "train", epochs=1, train_tuned=False)
                    
                    # Test predict with tuned model (use same experiment name as tuned training)
                    self.test_predict_with_existing_model(model, predict_tuned=True)
                    # Test predict with default model (use same experiment name as default training)
                    self.test_predict_with_existing_model(model, predict_tuned=False)
                elif mode == "train":
                    # Test train with both tuned and default parameters
                    self.test_mode(model, mode, epochs=1, n_trials=2, k_folds=2, train_tuned=True)  # Test tuned training
                    self.test_mode(model, mode, epochs=1, train_tuned=False)  # Test default training
                else:
                    self.test_mode(model, mode, epochs=1, n_trials=2, k_folds=2)
        
        # Test full workflow with one model
        self.test_full_workflow("LSTM")
        
        # Test report mode
        self.test_report_mode()
        
        end_time = datetime.now()
        self._generate_final_report(start_time, end_time)
    
    def _generate_final_report(self, start_time, end_time):
        """Generate final test report."""
        duration = end_time - start_time
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r['status'] == 'PASS'])
        failed_tests = len([r for r in self.test_results if r['status'] == 'FAIL'])
        warned_tests = len([r for r in self.test_results if r['status'] == 'WARN'])
        
        print("\n" + "=" * 80)
        print("📊 FINAL TEST REPORT")
        print("=" * 80)
        print(f"⏱️  Duration: {duration}")
        print(f"📈 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"⚠️  Warnings: {warned_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"📊 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if self.failed_tests:
            print(f"\n❌ Failed Tests:")
            for test in self.failed_tests:
                print(f"   - {test}")
        
        # Save detailed report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration.total_seconds(),
            'summary': {
                'total': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'warned': warned_tests,
                'success_rate': (passed_tests/total_tests)*100
            },
            'failed_tests': self.failed_tests,
            'detailed_results': self.test_results
        }
        
        report_file = self.base_dir / "Test" / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        print("=" * 80)
    
    def _cleanup(self):
        """Clean up test artifacts."""
        try:
            if self.temp_data_path and self.temp_data_path.exists():
                self.temp_data_path.unlink()
            
            # Clean up test directories
            test_dirs = [
                "Results/TEST_MODEL",
                "History/TEST_MODEL", 
                "Predictions/TEST_MODEL",
                "Metrics/TEST_MODEL",
                "Plots/TEST_MODEL",
                "Hyperparameters/TEST_MODEL",
                "Logs/TEST_MODEL",
                "Weights/TEST_MODEL"
            ]
            
            for dir_path in test_dirs:
                full_path = self.base_dir / dir_path
                if full_path.exists():
                    shutil.rmtree(full_path)
                    
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")


def main():
    """Main function to run the comprehensive test suite."""
    tester = TimeSeriesTester()
    
    try:
        tester.run_comprehensive_test()
    finally:
        tester._cleanup()


if __name__ == "__main__":
    main()