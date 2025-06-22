#!/usr/bin/env python3
"""
Comprehensive Test Script for Time Series Forecasting Package
Tests all models, modes, and functionality.

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


class TimeSeriesTester:
    """Comprehensive tester for the time series forecasting package."""
    
    def __init__(self):
        self.test_results = []
        self.failed_tests = []
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
    
    def test_mode(self, model: str, mode: str, epochs: int = 2, n_trials: int = 2):
        """Test a specific model and mode combination."""
        print(f"\n🧪 Testing {model} in {mode} mode...")
        
        try:
            # Build command
            cmd = [
                sys.executable, "main.py",
                "--model", model,
                "--data_name", "test_data",
                "--mode", mode,
                "--epochs", str(epochs),
                "--experiment_description", f"test_{mode}_{model.lower()}",
                "--sequence_length", "5"
            ]
            
            if mode == 'tune':
                cmd.extend(["--n_trials", str(n_trials)])
            
            # Run command
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Check if expected files were created
                expected_dirs = self._get_expected_directories(model, mode, f"test_{mode}_{model.lower()}")
                missing_dirs = []
                
                for dir_type, dir_path in expected_dirs.items():
                    if not dir_path.exists():
                        missing_dirs.append(dir_type)
                
                if missing_dirs:
                    self.log_test(f"{model} {mode}", "WARN", f"Missing directories: {missing_dirs}")
                else:
                    self.log_test(f"{model} {mode}", "PASS", "All expected files created")
                    
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                self.log_test(f"{model} {mode}", "FAIL", error_msg[:100])
                self.failed_tests.append(f"{model} {mode}")
                
        except subprocess.TimeoutExpired:
            self.log_test(f"{model} {mode}", "FAIL", "Test timed out after 5 minutes")
            self.failed_tests.append(f"{model} {mode}")
        except Exception as e:
            self.log_test(f"{model} {mode}", "FAIL", str(e))
            self.failed_tests.append(f"{model} {mode}")
    
    def _get_expected_directories(self, model: str, mode: str, experiment: str):
        """Get expected directory structure for a test."""
        base_dirs = {
            'results': self.base_dir / "Results" / model / mode / experiment,
            'history': self.base_dir / "History" / model / mode / experiment,
            'predictions': self.base_dir / "Predictions" / model / mode / experiment,
            'metrics': self.base_dir / "Metrics" / model / mode / experiment,
            'plots': self.base_dir / "Plots" / model / mode / experiment,
            'hyperparams': self.base_dir / "Hyperparameters" / model / experiment
        }
        
        if mode == 'tune':
            base_dirs['logs'] = self.base_dir / "Logs" / model
            
        return base_dirs
    
    def test_report_mode(self):
        """Test report mode functionality."""
        print("\n📊 Testing report mode...")
        
        try:
            cmd = [
                sys.executable, "main.py",
                "--model", "LSTM",
                "--data_name", "test_data",
                "--mode", "report",
                "--experiment_description", "test_apply_lstm"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Check if report contains expected information
                output = result.stdout
                if "Results" in output and ("LSTM" in output or "No results found" in output):
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
            result = subprocess.run([sys.executable, "main.py", "--help"], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                help_output = result.stdout
                required_elements = [
                    "--model", "--data_name", "--mode", 
                    "tune", "apply", "apply_not_tuned", "report",
                    "--experiment_description"
                ]
                
                missing_elements = [elem for elem in required_elements if elem not in help_output]
                
                if not missing_elements:
                    self.log_test("Help Message", "PASS", "All expected arguments present")
                else:
                    self.log_test("Help Message", "WARN", f"Missing elements: {missing_elements}")
            else:
                self.log_test("Help Message", "FAIL", "Help command failed")
                
        except Exception as e:
            self.log_test("Help Message", "FAIL", str(e))
    
    def test_data_processing(self):
        """Test data processing functionality."""
        print("\n🔄 Testing data processing...")
        
        try:
            from utils.data_utils import load_and_validate_data, get_data_path, prepare_data_loaders
            
            # Test data loading
            data_path = get_data_path("test_data")
            data, dates = load_and_validate_data(data_path)
            
            # Test data preparation
            train_loader, val_loader, test_loader, input_size = prepare_data_loaders(
                data, dates, sequence_length=5
            )
            
            # Verify shapes
            for batch_x, batch_y in train_loader:
                assert batch_x.shape[1] == 5  # sequence length
                assert batch_x.shape[2] == input_size  # features
                assert batch_y.shape[1] == 1  # target
                break
            
            self.log_test("Data Processing", "PASS", f"Input size: {input_size}, batch shapes verified")
            
        except Exception as e:
            self.log_test("Data Processing", "FAIL", str(e))
            self.failed_tests.append("Data Processing")
    
    def test_directory_creation(self):
        """Test robust directory creation."""
        print("\n📁 Testing directory creation...")
        
        try:
            from utils.file_utils import create_directory_safely, create_experiment_directories
            
            # Test basic directory creation
            test_dir = self.base_dir / "test_temp_dir"
            success = create_directory_safely(test_dir)
            
            if success and test_dir.exists():
                # Clean up
                test_dir.rmdir()
                
                # Test experiment directories
                dirs = create_experiment_directories("TestModel", "test", "test_experiment")
                
                missing_dirs = [name for name, path in dirs.items() if not path.exists()]
                
                if not missing_dirs:
                    self.log_test("Directory Creation", "PASS", f"Created {len(dirs)} directories")
                    # Clean up
                    self._cleanup_test_directories(dirs)
                else:
                    self.log_test("Directory Creation", "WARN", f"Missing: {missing_dirs}")
            else:
                self.log_test("Directory Creation", "FAIL", "Basic directory creation failed")
                
        except Exception as e:
            self.log_test("Directory Creation", "FAIL", str(e))
    
    def _cleanup_test_directories(self, dirs):
        """Clean up test directories."""
        try:
            for path in dirs.values():
                if path.exists() and path.is_dir():
                    # Remove files first
                    for file in path.rglob("*"):
                        if file.is_file():
                            file.unlink()
                    # Remove directories
                    for dir_path in sorted(path.rglob("*"), reverse=True):
                        if dir_path.is_dir():
                            dir_path.rmdir()
                    path.rmdir()
        except Exception:
            pass  # Ignore cleanup errors
    
    def run_comprehensive_test(self):
        """Run all tests."""
        print("🚀 Starting Comprehensive Time Series Forecasting Tests")
        print("=" * 60)
        
        start_time = datetime.now()
        
        # Create test data
        self.create_test_data()
        
        # Test basic functionality
        self.test_help_message()
        self.test_models_available()
        self.test_data_processing()
        self.test_directory_creation()
        
        # Test core functionality with one model
        print("\n🔥 Testing core functionality (LSTM)...")
        self.test_mode("LSTM", "apply_not_tuned", epochs=2)
        self.test_mode("LSTM", "apply", epochs=2)
        self.test_mode("LSTM", "tune", epochs=2, n_trials=2)
        
        # Test report mode
        self.test_report_mode()
        
        # Quick test with other models (just apply_not_tuned mode)
        other_models = ["TCN", "Transformer", "MLP"]
        for model in other_models:
            self.test_mode(model, "apply_not_tuned", epochs=2)
        
        # Generate report
        end_time = datetime.now()
        self._generate_final_report(start_time, end_time)
    
    def _generate_final_report(self, start_time, end_time):
        """Generate final test report."""
        print("\n" + "=" * 60)
        print("📋 FINAL TEST REPORT")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r['status'] == 'PASS'])
        failed_tests = len([r for r in self.test_results if r['status'] == 'FAIL'])
        warning_tests = len([r for r in self.test_results if r['status'] == 'WARN'])
        
        print(f"⏱️  Total Time: {end_time - start_time}")
        print(f"📊 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"⚠️  Warnings: {warning_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"📈 Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        if self.failed_tests:
            print(f"\n❌ Failed Tests:")
            for test in self.failed_tests:
                print(f"   • {test}")
        
        # Save detailed report
        report_path = self.base_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump({
                'summary': {
                    'total_tests': total_tests,
                    'passed': passed_tests,
                    'failed': failed_tests,
                    'warnings': warning_tests,
                    'success_rate': passed_tests/total_tests*100,
                    'duration': str(end_time - start_time),
                    'timestamp': datetime.now().isoformat()
                },
                'detailed_results': self.test_results,
                'failed_tests': self.failed_tests
            }, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_path}")
        
        # Cleanup
        self._cleanup()
        
        if failed_tests == 0:
            print("\n🎉 All tests passed! The system is working correctly.")
        else:
            print(f"\n⚠️  {failed_tests} tests failed. Please check the issues above.")
    
    def _cleanup(self):
        """Clean up test files."""
        try:
            if self.temp_data_path and self.temp_data_path.exists():
                self.temp_data_path.unlink()
                
            # Clean up test experiment directories
            cleanup_patterns = ["test_*", "TestModel"]
            for pattern in cleanup_patterns:
                for base_dir in ["Results", "History", "Predictions", "Metrics", "Plots", "Hyperparameters", "Logs"]:
                    base_path = Path(base_dir)
                    if base_path.exists():
                        for item in base_path.rglob(pattern):
                            if item.is_dir():
                                shutil.rmtree(item, ignore_errors=True)
                            elif item.is_file():
                                item.unlink(missing_ok=True)
        except Exception:
            pass  # Ignore cleanup errors


def main():
    """Main function to run tests."""
    tester = TimeSeriesTester()
    tester.run_comprehensive_test()


if __name__ == "__main__":
    main() 