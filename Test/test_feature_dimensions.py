"""
Feature Dimension Testing Script
Tests the Time Series Forecasting system with different numbers of features.

Author: Soroosh Shalileh
Email: sr.shalileh@gmail.com
Year: 2025
"""

import pandas as pd
import numpy as np
import subprocess
import sys
from pathlib import Path
import json
from datetime import datetime


class FeatureDimensionTester:
    """Test the system with different feature dimensions."""
    
    def __init__(self):
        self.base_dir = Path.cwd()
        self.test_results = []
        self.failed_tests = []
        
    def create_test_data(self, n_features: int, filename: str) -> Path:
        """Create synthetic test data with specified number of features."""
        print(f"\n📊 Creating test data with {n_features} features...")
        
        # Create date range
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='h')
        
        # Create synthetic features
        data = np.random.randn(200, n_features).cumsum(axis=0)
        
        # Add some trend and seasonality to first feature
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
        temp_file = data_dir / f"{filename}.csv"
        df.to_csv(temp_file, index=False)
        
        print(f"✅ Created test data: {len(df)} rows with {n_features} features")
        return temp_file
    
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
    
    def test_model_with_features(self, model: str, n_features: int, sequence_length: int = 5):
        """Test a model with specific number of features."""
        data_name = f"test_{n_features}feat"
        test_name = f"{model} with {n_features} features"
        
        print(f"\n🧪 Testing {test_name}...")
        
        try:
            # Create test data
            self.create_test_data(n_features, data_name)
            
            # Build command - using train mode with default parameters
            cmd = [
                sys.executable, "main.py",
                "--model", model,
                "--data_name", data_name,
                "--mode", "train",
                "--train_tuned", "false",
                "--epochs", "2",
                "--experiment_description", f"test_{n_features}feat_{model.lower()}",
                "--sequence_length", str(sequence_length)
            ]
            
            # Run command
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                # Check if expected weight files were created
                weights_dir = self.base_dir / "Weights"
                expected_weight_file = f"{model}_{data_name}_test_{n_features}feat_{model.lower()}_{sequence_length}_default_best.pth"
                weight_file_path = weights_dir / expected_weight_file
                
                if weight_file_path.exists():
                    self.log_test(test_name, "PASS", f"Successfully processed {n_features} features")
                else:
                    self.log_test(test_name, "WARN", f"Completed but missing expected weight file: {expected_weight_file}")
            else:
                error_msg = result.stderr.strip() if result.stderr else result.stdout.strip()
                self.log_test(test_name, "FAIL", error_msg[:100] + "..." if len(error_msg) > 100 else error_msg)
                self.failed_tests.append(test_name)
                
        except subprocess.TimeoutExpired:
            self.log_test(test_name, "FAIL", "Test timed out after 2 minutes")
            self.failed_tests.append(test_name)
        except Exception as e:
            self.log_test(test_name, "FAIL", str(e))
            self.failed_tests.append(test_name)
    
    def test_sequence_length_variations(self, model: str = "LSTM", n_features: int = 3):
        """Test different sequence lengths with a specific model."""
        print(f"\n🔄 Testing sequence length variations with {model}...")
        
        sequence_lengths = [3, 5, 10, 15, 20]
        
        for seq_len in sequence_lengths:
            data_name = f"test_seq{seq_len}"
            test_name = f"{model} with seq_len={seq_len}"
            
            print(f"\n🧪 Testing {test_name}...")
            
            try:
                # Create test data
                self.create_test_data(n_features, data_name)
                
                # Build command - using train mode with default parameters
                cmd = [
                    sys.executable, "main.py",
                    "--model", model,
                    "--data_name", data_name,
                    "--mode", "train",
                    "--train_tuned", "false",
                    "--epochs", "2",
                    "--experiment_description", f"test_seq{seq_len}_{model.lower()}",
                    "--sequence_length", str(seq_len)
                ]
                
                # Run command
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    self.log_test(test_name, "PASS", f"Successfully processed seq_len={seq_len}")
                else:
                    error_msg = result.stderr.strip() if result.stderr else result.stdout.strip()
                    self.log_test(test_name, "FAIL", error_msg[:100] + "..." if len(error_msg) > 100 else error_msg)
                    self.failed_tests.append(test_name)
                    
            except subprocess.TimeoutExpired:
                self.log_test(test_name, "FAIL", "Test timed out after 2 minutes")
                self.failed_tests.append(test_name)
            except Exception as e:
                self.log_test(test_name, "FAIL", str(e))
                self.failed_tests.append(test_name)
    
    def run_comprehensive_feature_test(self):
        """Run comprehensive tests with different feature dimensions."""
        print("🚀 Starting Feature Dimension Testing")
        print("=" * 60)
        
        start_time = datetime.now()
        
        # Test different feature dimensions
        feature_dimensions = [1, 3, 5, 10, 20]
        models_to_test = ["LSTM", "MLP", "TCN"]
        
        print(f"\n🔢 Testing {len(feature_dimensions)} feature dimensions with {len(models_to_test)} models...")
        
        for n_features in feature_dimensions:
            print(f"\n📊 Testing with {n_features} features:")
            for model in models_to_test:
                self.test_model_with_features(model, n_features)
        
        # Test sequence length variations
        self.test_sequence_length_variations()
        
        # Test edge cases
        print(f"\n🧪 Testing edge cases...")
        
        # Very small dataset
        try:
            self.create_test_data_small()
            self.test_model_with_features("LSTM", 3, sequence_length=3)
        except Exception as e:
            self.log_test("Small dataset test", "FAIL", str(e))
        
        # Generate report
        end_time = datetime.now()
        self._generate_final_report(start_time, end_time)
    
    def create_test_data_small(self, filename: str = "test_small"):
        """Create a very small test dataset."""
        print(f"\n📊 Creating small test data...")
        
        # Create small date range
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=50, freq='h')
        
        # Create synthetic features
        data = np.random.randn(50, 3).cumsum(axis=0)
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(3)])
        df['date'] = dates
        df = df[['date'] + [f'feature_{i}' for i in range(3)]]
        
        # Save to data directory
        data_dir = self.base_dir / "data"
        data_dir.mkdir(exist_ok=True)
        temp_file = data_dir / f"{filename}.csv"
        df.to_csv(temp_file, index=False)
        
        print(f"✅ Created small test data: {len(df)} rows with 3 features")
        return temp_file
    
    def _generate_final_report(self, start_time, end_time):
        """Generate final test report."""
        print("\n" + "=" * 60)
        print("📋 FEATURE DIMENSION TEST REPORT")
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
        report_path = self.base_dir / f"feature_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
        
        if failed_tests == 0:
            print("\n🎉 All feature dimension tests passed! The system is robust across different data shapes.")
        else:
            print(f"\n⚠️  {failed_tests} tests failed. Please check the issues above.")
        
        # Cleanup test files
        self._cleanup()
    
    def _cleanup(self):
        """Clean up test files."""
        try:
            # Clean up test data files
            data_dir = Path("data")
            for pattern in ["test_*feat.csv", "test_seq*.csv", "test_small.csv"]:
                for file in data_dir.glob(pattern):
                    file.unlink(missing_ok=True)
            
            # Clean up test experiment directories
            cleanup_patterns = ["test_*feat_*", "test_seq*_*", "test_small_*"]
            for pattern in cleanup_patterns:
                for base_dir in ["Results", "History", "Predictions", "Metrics", "Plots", "Hyperparameters"]:
                    base_path = Path(base_dir)
                    if base_path.exists():
                        for item in base_path.rglob(pattern):
                            if item.is_dir():
                                import shutil
                                shutil.rmtree(item, ignore_errors=True)
                            elif item.is_file():
                                item.unlink(missing_ok=True)
        except Exception:
            pass  # Ignore cleanup errors


def main():
    """Main function to run feature dimension tests."""
    tester = FeatureDimensionTester()
    tester.run_comprehensive_feature_test()


if __name__ == "__main__":
    main() 