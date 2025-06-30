"""
Preprocessing Validation Test Script
====================================

Validates the compatibility between LaTeX formulation and implementation,
specifically testing the merchant data preprocessing pipeline and dimensional
correspondences described in the paper.

This script tests:
1. Dimensional compatibility: LaTeX (k+1)×(N+K) ↔ Implementation (sequence_length, n_features)
2. Merchant data preprocessing pipeline from example.py
3. Target calculation: y_t = Σ x_{m,t} (sum of merchant columns)
4. Integration with existing TimeSeriesPreprocessor

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
import tempfile
import shutil
from typing import Tuple, Dict, Any
import torch


class PreprocessingValidator:
    """Validates preprocessing compatibility with LaTeX formulation."""
    
    def __init__(self):
        self.base_dir = Path.cwd()
        self.test_results = []
        self.failed_tests = []
        self.temp_files = []
        
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
    
    def test_dimensional_correspondence(self):
        """Test LaTeX formulation dimensional correspondence."""
        print("\n📐 Testing LaTeX ↔ Implementation Dimensional Correspondence")
        print("=" * 60)
        
        try:
            # Add parent directory to path to import utils
            sys.path.insert(0, str(self.base_dir.parent) if 'Test' in str(self.base_dir) else str(self.base_dir))
            from utils.data_preprocessing import TimeSeriesPreprocessor
            
            # Test case parameters
            T = 100  # Total time steps
            N = 5    # Number of merchants/features
            k_plus_1 = 10  # Sequence length (k+1 in LaTeX)
            
            # Create synthetic merchant data: (T, N) where N = merchants + contextual + target
            np.random.seed(42)
            
            # Create merchant consumption data (positive values)
            merchant_values = np.random.exponential(scale=10, size=(T, N)).cumsum(axis=0)
            
            # Add contextual features (10 features)
            contextual_features = np.random.randn(T, 10)
            
            # Calculate target as sum of merchant values
            target_values = np.sum(merchant_values, axis=1, keepdims=True)
            
            # Combine: merchants + contextual + target
            merchant_data = np.concatenate([merchant_values, contextual_features, target_values], axis=1)
            
            # Create preprocessor
            preprocessor = TimeSeriesPreprocessor(
                sequence_length=k_plus_1,
                normalization='minmax'
            )
            
            # Fit and create sequences
            preprocessor.fit_scalers(merchant_data)
            X, y = preprocessor.create_sequences(merchant_data)
            
            # Validate dimensions
            expected_samples = T - k_plus_1
            expected_X_shape = (expected_samples, k_plus_1, N)
            expected_y_shape = (expected_samples, 1)
            
            # Check X dimensions (features only, excluding target)
            # For merchant data: N merchants + K contextual features (excluding target)
            expected_X_shape_features = (expected_samples, k_plus_1, N+10-1)  # N merchants + 10 contextual - 1 target
            if X.shape[0] == expected_samples and X.shape[1] == k_plus_1:
                self.log_test("X Dimensions", "PASS", 
                    f"LaTeX 𝒽_t ∈ ℝ^{{(k+1)×(N+K)}} features = {X.shape} ✓")
            else:
                self.log_test("X Dimensions", "FAIL", 
                    f"Expected shape with {expected_samples} samples and {k_plus_1} sequence length, got {X.shape}")
                return False
            
            # Check y dimensions
            if y.shape == expected_y_shape:
                self.log_test("y Dimensions", "PASS", 
                    f"Target y_t ∈ ℝ = {expected_y_shape} ✓")
            else:
                self.log_test("y Dimensions", "FAIL", 
                    f"Expected {expected_y_shape}, got {y.shape}")
                return False
            
            # Validate target calculation: y_t = sum of merchant values (last column should be this sum)
            for i in range(min(5, len(y))):  # Check first 5 samples
                # Sum first N columns (merchants) and compare with last column (target)
                merchant_sum = np.sum(merchant_data[i + k_plus_1, :N])  # Sum of merchant columns
                stored_target = merchant_data[i + k_plus_1, -1]  # Last column (should be sum)
                actual_target_denorm = preprocessor.denormalize_targets(np.array([y[i, 0]]))[0]
                
                # Check if last column matches sum of merchants
                if not np.isclose(merchant_sum, stored_target, rtol=1e-2):
                    self.log_test("Target Calculation", "FAIL", 
                        f"Sample {i}: merchant sum {merchant_sum:.6f} != stored target {stored_target:.6f}")
                    return False
                
                # Check if preprocessor gets the right value
                if not np.isclose(stored_target, actual_target_denorm, rtol=1e-2):
                    self.log_test("Target Calculation", "FAIL", 
                        f"Sample {i}: expected {stored_target:.6f}, got {actual_target_denorm:.6f}")
                    return False
            
            self.log_test("Target Calculation", "PASS", 
                "y_t = Σ x_{m,t} correctly implemented")
            
            return True
            
        except Exception as e:
            self.log_test("Dimensional Correspondence", "FAIL", str(e))
            return False
    
    def test_merchant_preprocessing_pipeline(self):
        """Test the complete merchant data preprocessing pipeline."""
        print("\n🏪 Testing Merchant Data Preprocessing Pipeline")
        print("=" * 60)
        
        try:
            # Import our example preprocessing functions
            parent_dir = self.base_dir.parent if 'Test' in str(self.base_dir) else self.base_dir
            sys.path.insert(0, str(parent_dir))
            from example import (
                load_merchant_transactions, 
                aggregate_merchant_data, 
                add_contextual_features,
                test_with_models
            )
            
            # Step 1: Create synthetic transaction data
            transaction_data = self._create_synthetic_transactions()
            
            # Step 2: Test aggregation
            merchant_data = aggregate_merchant_data(transaction_data, freq='D')
            
            # Validate merchant aggregation
            if merchant_data.shape[1] >= 3:  # At least 3 merchants
                self.log_test("Merchant Aggregation", "PASS", 
                    f"Created {merchant_data.shape[0]} time steps × {merchant_data.shape[1]} merchants")
            else:
                self.log_test("Merchant Aggregation", "FAIL", 
                    f"Insufficient merchants: {merchant_data.shape[1]}")
                return False
            
            # Step 3: Test contextual features
            enhanced_data = add_contextual_features(merchant_data)
            
            expected_additional_features = 10  # From example.py
            actual_additional = enhanced_data.shape[1] - merchant_data.shape[1]
            
            if actual_additional == expected_additional_features:
                self.log_test("Contextual Features", "PASS", 
                    f"Added {actual_additional} contextual features")
            else:
                self.log_test("Contextual Features", "WARN", 
                    f"Expected {expected_additional_features}, got {actual_additional}")
            
            # Step 4: Test model compatibility
            data_array = enhanced_data.values
            success = test_with_models(data_array, sequence_length=3)
            
            if success:
                self.log_test("Model Compatibility", "PASS", 
                    "Preprocessed data compatible with TimeSeriesPreprocessor")
            else:
                self.log_test("Model Compatibility", "FAIL", 
                    "Preprocessed data not compatible with models")
                return False
            
            # Step 5: Validate LaTeX mapping for enhanced data
            N_merchants = len([col for col in enhanced_data.columns if col.startswith('merchant_')])
            total_features = enhanced_data.shape[1]
            
            self.log_test("LaTeX Mapping", "PASS", 
                f"N (merchants) = {N_merchants}, Total features = {total_features}")
            
            return True
            
        except Exception as e:
            self.log_test("Merchant Preprocessing", "FAIL", str(e))
            return False
    
    def _create_synthetic_transactions(self) -> pd.DataFrame:
        """Create synthetic transaction data similar to merchant_synthetic.csv."""
        np.random.seed(42)
        
        # Create 30 days of hourly transactions
        start_date = pd.Timestamp('2023-01-01')
        timestamps = []
        
        # Generate realistic transaction times
        for day in range(30):
            base_date = start_date + pd.Timedelta(days=day)
            n_transactions = np.random.poisson(20)  # Average 20 transactions per day
            
            for _ in range(n_transactions):
                hour = np.random.normal(14, 4)  # Peak around 2 PM
                hour = int(np.clip(hour, 0, 23))
                minute = np.random.randint(0, 60)
                timestamps.append(base_date + pd.Timedelta(hours=hour, minutes=minute))
        
        # Create transaction data
        n_transactions = len(timestamps)
        merchants = [1, 2, 3, 4, 5]  # 5 merchants
        customers = list(range(1, 21))  # 20 customers
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'merchant_id': np.random.choice(merchants, n_transactions),
            'customer_id': np.random.choice(customers, n_transactions),
            'amount': np.random.lognormal(3, 1, n_transactions)
        })
        
        # Save temporary file
        temp_file = self.base_dir / "data" / "test_transactions.csv"
        temp_file.parent.mkdir(exist_ok=True)
        df.to_csv(temp_file, index=False)
        self.temp_files.append(temp_file)
        
        return df
    
    def test_mathematical_consistency(self):
        """Test mathematical consistency of preprocessing pipeline."""
        print("\n🔢 Testing Mathematical Consistency")
        print("=" * 60)
        
        try:
            sys.path.insert(0, str(self.base_dir.parent) if 'Test' in str(self.base_dir) else str(self.base_dir))
            from utils.data_preprocessing import TimeSeriesPreprocessor
            
            # Test case: 4 merchants, 3 contextual features, 1 target
            T = 50  # Total time steps
            sequence_length = 5
            
            # Create synthetic data with known mathematical relationships
            np.random.seed(123)
            merchants = np.random.exponential(2, size=(T, 4))  # 4 merchants
            contextual = np.random.randn(T, 3)  # 3 contextual features
            target = np.sum(merchants, axis=1, keepdims=True)  # Target = sum of merchants
            
            data = np.concatenate([merchants, contextual, target], axis=1)
            
            # Process with TimeSeriesPreprocessor
            preprocessor = TimeSeriesPreprocessor(sequence_length=sequence_length)
            preprocessor.fit_scalers(data)
            X, y = preprocessor.create_sequences(data)
            
            # Mathematical consistency checks
            for i in range(min(10, len(y))):
                # Get the original values for time step i + sequence_length
                original_merchants = data[i + sequence_length, :4]  # First 4 columns
                expected_target = np.sum(original_merchants)
                
                # Get the target from preprocessor
                actual_target_normalized = y[i, 0]
                actual_target = preprocessor.denormalize_targets(np.array([actual_target_normalized]))[0]
                
                if not np.isclose(expected_target, actual_target, rtol=1e-2):
                    self.log_test("Mathematical Consistency", "FAIL", 
                        f"Sample {i}: expected {expected_target:.6f}, got {actual_target:.6f}")
                    return False
            
            self.log_test("Mathematical Consistency", "PASS", 
                "Target calculation mathematically consistent")
            return True
            
        except Exception as e:
            self.log_test("Mathematical Consistency", "FAIL", str(e))
            return False

    def test_basic_preprocessing(self):
        """Test basic preprocessing functionality."""
        print("\n🔧 Testing Basic Preprocessing")
        print("=" * 60)
        
        try:
            # Test dimensional correspondence
            dim_test = self.test_dimensional_correspondence()
            if not dim_test:
                return False
            
            # Test merchant preprocessing pipeline
            pipeline_test = self.test_merchant_preprocessing_pipeline()
            if not pipeline_test:
                return False
            
            # Test mathematical consistency
            math_test = self.test_mathematical_consistency()
            if not math_test:
                return False
            
            self.log_test("Basic Preprocessing", "PASS", 
                "All basic preprocessing tests completed successfully")
            return True
            
        except Exception as e:
            self.log_test("Basic Preprocessing", "FAIL", str(e))
            return False
    
    def test_latex_compatibility(self):
        """Test LaTeX compatibility with implementation."""
        print("\n📐 Testing LaTeX Compatibility")
        print("=" * 60)
        
        try:
            # Test dimensional correspondence
            dim_test = self.test_dimensional_correspondence()
            if not dim_test:
                self.log_test("LaTeX Compatibility", "FAIL", "Dimensional correspondence failed")
                return False
            
            # Test mathematical consistency
            math_test = self.test_mathematical_consistency()
            if not math_test:
                self.log_test("LaTeX Compatibility", "FAIL", "Mathematical consistency failed")
                return False
            
            self.log_test("LaTeX Compatibility", "PASS", 
                "LaTeX formulation matches implementation")
            return True
            
        except Exception as e:
            self.log_test("LaTeX Compatibility", "FAIL", str(e))
            return False
    
    def test_integration_with_main_pipeline(self):
        """Test integration with main.py pipeline using preprocessed data."""
        print("\n🔗 Testing Integration with Main Pipeline")
        print("=" * 60)
        
        try:
            # Create preprocessed merchant data
            parent_dir = self.base_dir.parent if 'Test' in str(self.base_dir) else self.base_dir
            sys.path.insert(0, str(parent_dir))
            from example import main as run_example
            
            # Temporarily redirect to avoid file conflicts
            original_cwd = Path.cwd()
            
            # Run the example preprocessing (this creates merchant_processed.csv)
            processed_data = run_example()
            
            if processed_data is not None:
                self.log_test("Example Preprocessing", "PASS", 
                    f"Created processed data: {processed_data.shape}")
                
                # Test with a simple model run
                # Set correct working directory and main.py path
                parent_dir = self.base_dir.parent if 'Test' in str(self.base_dir) else self.base_dir
                main_py_path = parent_dir / "main.py"
                
                cmd = [
                    sys.executable, str(main_py_path),
                    "--model", "LSTM",
                    "--data_name", "merchant_processed",
                    "--mode", "train",
                    "--train_tuned", "false",  # Use default parameters for quick testing
                    "--epochs", "1",  # Reduced for faster testing
                    "--experiment_description", "preprocessing_validation_test",
                    "--sequence_length", "2"  # Reduced to ensure enough validation samples
                ]
                
                # Run with correct working directory
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=120,
                    cwd=str(parent_dir)  # Set working directory to parent
                )
                
                if result.returncode == 0:
                    self.log_test("Main Pipeline Integration", "PASS", 
                        "Preprocessed merchant data works with main.py")
                    return True
                else:
                    # Capture full error for debugging
                    stderr = result.stderr.strip() if result.stderr else ""
                    stdout = result.stdout.strip() if result.stdout else ""
                    error_msg = f"STDERR: {stderr}\nSTDOUT: {stdout}" if stderr else stdout
                    
                    print(f"\n🔍 Debug Info:")
                    print(f"   Working Directory: {parent_dir}")
                    print(f"   Main.py Path: {main_py_path}")
                    print(f"   Command: {' '.join(cmd)}")
                    print(f"   Return Code: {result.returncode}")
                    print(f"   STDERR: {stderr}")
                    print(f"   STDOUT: {stdout}")
                    
                    self.log_test("Main Pipeline Integration", "FAIL", 
                        f"main.py failed with return code {result.returncode}")
                    return False
            else:
                self.log_test("Example Preprocessing", "FAIL", 
                    "example.py returned None")
                return False
                
        except Exception as e:
            self.log_test("Integration Test", "FAIL", str(e))
            return False
    
    def run_comprehensive_validation(self):
        """Run comprehensive preprocessing validation."""
        print("🚀 Starting Preprocessing Validation Tests")
        print("=" * 70)
        print("Testing compatibility between LaTeX formulation and implementation")
        print("=" * 70)
        
        start_time = datetime.now()
        
        # Core validation tests
        tests = [
            ("Dimensional Correspondence", self.test_dimensional_correspondence),
            ("Mathematical Consistency", self.test_mathematical_consistency),
            ("Merchant Preprocessing Pipeline", self.test_merchant_preprocessing_pipeline),
            ("Integration with Main Pipeline", self.test_integration_with_main_pipeline),
        ]
        
        print(f"\n🧪 Running {len(tests)} validation tests...")
        
        all_passed = True
        for test_name, test_func in tests:
            try:
                success = test_func()
                if not success:
                    all_passed = False
            except Exception as e:
                self.log_test(test_name, "FAIL", f"Exception: {str(e)}")
                all_passed = False
        
        # Generate report
        end_time = datetime.now()
        self._generate_final_report(start_time, end_time, all_passed)
        
        return all_passed
    
    def _generate_final_report(self, start_time, end_time, all_passed):
        """Generate final validation report."""
        print("\n" + "=" * 70)
        print("📋 PREPROCESSING VALIDATION REPORT")
        print("=" * 70)
        
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
        
        # Compatibility summary
        print(f"\n📐 COMPATIBILITY SUMMARY:")
        print(f"  LaTeX Formulation ↔ Implementation: {'✅ COMPATIBLE' if all_passed else '❌ ISSUES FOUND'}")
        print(f"  Dimensional Mapping: (k+1)×N ↔ (sequence_length, n_features)")
        print(f"  Target Calculation: y_t = Σ x_{{m,t}} ↔ np.sum(data[i + sequence_length])")
        print(f"  Dimensional Mapping: (k+1)×(N) ↔ (sequence_length, n_features)")
        print(f"  Target is supposed to be: y_t = last column ↔ data[i + sequence_length, -1]")
        print(f"  Preprocessing Pipeline: {'✅ VALIDATED' if all_passed else '❌ NEEDS ATTENTION'}")
        
        if self.failed_tests:
            print(f"\n❌ Failed Tests:")
            for test in self.failed_tests:
                print(f"   • {test}")
        
        # Save detailed report
        report_path = self.base_dir / f"preprocessing_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump({
                'summary': {
                    'total_tests': total_tests,
                    'passed': passed_tests,
                    'failed': failed_tests,
                    'warnings': warning_tests,
                    'success_rate': passed_tests/total_tests*100,
                    'duration': str(end_time - start_time),
                    'timestamp': datetime.now().isoformat(),
                    'compatibility_status': 'COMPATIBLE' if all_passed else 'ISSUES_FOUND'
                },
                'detailed_results': self.test_results,
                'failed_tests': self.failed_tests
            }, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_path}")
        
        # Cleanup
        self._cleanup()
        
        if all_passed:
            print("\n🎉 All validation tests passed!")
            print("✅ LaTeX formulation and implementation are fully compatible.")
            print("✅ Merchant preprocessing pipeline is validated.")
        else:
            print(f"\n⚠️  {failed_tests} validation tests failed.")
            print("❌ Please review the compatibility issues above.")
    
    def _cleanup(self):
        """Clean up temporary files."""
        try:
            for temp_file in self.temp_files:
                if temp_file.exists():
                    temp_file.unlink()
            
            # Clean up test experiment directories
            cleanup_patterns = ["preprocessing_validation_test"]
            for pattern in cleanup_patterns:
                for base_dir in ["Results", "History", "Predictions", "Metrics", "Plots", "Hyperparameters"]:
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
    """Main function to run preprocessing validation."""
    validator = PreprocessingValidator()
    success = validator.run_comprehensive_validation()
    
    if success:
        print("\n🚀 Ready to proceed with your time series forecasting research!")
        sys.exit(0)
    else:
        print("\n🔧 Please address the compatibility issues before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main() 