#!/usr/bin/env python3
"""
Pytest-compatible test file for TimeCast
Tests the new three-mode system (tune, train, predict) with train_tuned and predict_tuned options.

Author: Soroosh Shalileh
Email: sr.shalileh@gmail.com
Year: 2025
"""

import pytest
import sys
from pathlib import Path

# Add the root directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from Test.test_script import TimeSeriesTester
from Test.test_feature_dimensions import FeatureDimensionTester
from Test.test_preprocessing_validation import PreprocessingValidator


@pytest.fixture(scope="session")
def time_series_tester():
    """Create a TimeSeriesTester instance for the test session."""
    tester = TimeSeriesTester()
    tester.create_test_data()
    yield tester
    # Cleanup after tests
    tester._cleanup()


@pytest.fixture(scope="session") 
def feature_tester():
    """Create a FeatureDimensionTester instance for the test session."""
    return FeatureDimensionTester()


@pytest.fixture(scope="session")
def preprocessing_validator():
    """Create a PreprocessingValidator instance for the test session."""
    return PreprocessingValidator()


def test_models_available(time_series_tester):
    """Test that all models can be imported."""
    time_series_tester.test_models_available()
    # Check if any model tests failed
    model_failures = [test for test in time_series_tester.failed_tests if 'Model' in test]
    assert len(model_failures) == 0, f"Model import failures: {model_failures}"


def test_help_message(time_series_tester):
    """Test that help message works."""
    time_series_tester.test_help_message()
    help_failures = [test for test in time_series_tester.failed_tests if 'Help' in test]
    assert len(help_failures) == 0, f"Help message failures: {help_failures}"


def test_data_processing(time_series_tester):
    """Test data processing functionality."""
    time_series_tester.test_data_processing()
    data_failures = [test for test in time_series_tester.failed_tests if 'Data Processing' in test]
    assert len(data_failures) == 0, f"Data processing failures: {data_failures}"


def test_unique_specifier_system(time_series_tester):
    """Test the unique specifier system."""
    time_series_tester.test_unique_specifier_system()
    specifier_failures = [test for test in time_series_tester.failed_tests if 'Unique Specifier' in test]
    assert len(specifier_failures) == 0, f"Unique specifier failures: {specifier_failures}"


# Test individual modes
@pytest.mark.slow
def test_lstm_tune_mode(time_series_tester):
    """Test LSTM in tune mode."""
    time_series_tester.test_mode("LSTM", "tune", epochs=2, n_trials=3)
    tune_failures = [test for test in time_series_tester.failed_tests if 'LSTM tune' in test]
    assert len(tune_failures) == 0, f"LSTM tune failures: {tune_failures}"


@pytest.mark.slow
def test_lstm_train_default_mode(time_series_tester):
    """Test LSTM in train mode with default parameters."""
    time_series_tester.test_mode("LSTM", "train", epochs=2, train_tuned=False)
    train_default_failures = [test for test in time_series_tester.failed_tests if 'LSTM train (default)' in test]
    assert len(train_default_failures) == 0, f"LSTM train default failures: {train_default_failures}"


@pytest.mark.slow  
def test_lstm_train_mode(time_series_tester):
    """Test LSTM in train mode (requires tune mode first)."""
    # First run tune mode to get hyperparameters
    time_series_tester.test_mode("LSTM", "tune", epochs=2, n_trials=2)
    # Then run train mode
    time_series_tester.test_mode("LSTM", "train", epochs=2, k_folds=2)
    train_failures = [test for test in time_series_tester.failed_tests if 'LSTM train (tuned)' in test]
    assert len(train_failures) == 0, f"LSTM train failures: {train_failures}"


@pytest.mark.slow
def test_lstm_predict_mode(time_series_tester):
    """Test LSTM in predict mode (requires trained models)."""
    # First ensure we have trained models
    time_series_tester.test_mode("LSTM", "train", epochs=2, train_tuned=False)
    # Test predict with default model
    time_series_tester.test_mode("LSTM", "predict", epochs=1, predict_tuned=False)  # epochs not used in predict mode
    predict_failures = [test for test in time_series_tester.failed_tests if 'LSTM predict' in test]
    assert len(predict_failures) == 0, f"LSTM predict failures: {predict_failures}"


@pytest.mark.slow
def test_mlp_train_default_mode(time_series_tester):
    """Test MLP in train mode with default parameters."""
    time_series_tester.test_mode("MLP", "train", epochs=2, train_tuned=False)
    mlp_failures = [test for test in time_series_tester.failed_tests if 'MLP train (default)' in test]
    assert len(mlp_failures) == 0, f"MLP train default failures: {mlp_failures}"


@pytest.mark.slow
def test_tcn_train_default_mode(time_series_tester):
    """Test TCN in train mode with default parameters."""
    time_series_tester.test_mode("TCN", "train", epochs=2, train_tuned=False)
    tcn_failures = [test for test in time_series_tester.failed_tests if 'TCN train (default)' in test]
    assert len(tcn_failures) == 0, f"TCN train default failures: {tcn_failures}"


def test_report_mode(time_series_tester):
    """Test report mode functionality."""
    time_series_tester.test_report_mode()
    report_failures = [test for test in time_series_tester.failed_tests if 'Report Mode' in test]
    assert len(report_failures) == 0, f"Report mode failures: {report_failures}"


@pytest.mark.integration
def test_full_workflow_lstm(time_series_tester):
    """Test the complete four-mode workflow for LSTM."""
    time_series_tester.test_full_workflow("LSTM")
    workflow_failures = [test for test in time_series_tester.failed_tests if 'Full Workflow LSTM' in test]
    assert len(workflow_failures) == 0, f"Full workflow failures: {workflow_failures}"


@pytest.mark.integration
def test_feature_dimensions(feature_tester):
    """Test feature dimensions with different models."""
    feature_tester.test_model_with_features("LSTM", 3, sequence_length=5)
    assert len(feature_tester.failed_tests) == 0, f"Feature dimension failures: {feature_tester.failed_tests}"


@pytest.mark.integration
def test_preprocessing_validation(preprocessing_validator):
    """Test preprocessing validation."""
    preprocessing_validator.test_basic_preprocessing()
    preprocessing_validator.test_latex_compatibility()
    
    # Check for failures
    validation_failures = [result for result in preprocessing_validator.test_results 
                          if result.get('status') == 'FAIL']
    assert len(validation_failures) == 0, f"Preprocessing validation failures: {validation_failures}"


# Quick smoke tests (run by default)
def test_smoke_lstm_train_default(time_series_tester):
    """Quick smoke test for LSTM train mode with default parameters."""
    time_series_tester.test_mode("LSTM", "train", epochs=1, train_tuned=False)
    lstm_failures = [test for test in time_series_tester.failed_tests if 'LSTM train (default)' in test]
    assert len(lstm_failures) == 0, f"LSTM smoke test failures: {lstm_failures}"


def test_smoke_mlp_train_default(time_series_tester):
    """Quick smoke test for MLP train mode with default parameters."""
    time_series_tester.test_mode("MLP", "train", epochs=1, train_tuned=False)
    mlp_failures = [test for test in time_series_tester.failed_tests if 'MLP train (default)' in test]
    assert len(mlp_failures) == 0, f"MLP smoke test failures: {mlp_failures}"


def test_smoke_tune_mode(time_series_tester):
    """Quick smoke test for tune mode."""
    time_series_tester.test_mode("LSTM", "tune", epochs=1, n_trials=2)
    tune_failures = [test for test in time_series_tester.failed_tests if 'LSTM tune' in test]
    assert len(tune_failures) == 0, f"Tune mode smoke test failures: {tune_failures}"


# Test backward compatibility with old modes (if still supported)
@pytest.mark.slow
@pytest.mark.legacy
def test_legacy_apply_not_tuned(time_series_tester):
    """Test legacy apply_not_tuned mode (if still supported)."""
    try:
        time_series_tester.test_mode("LSTM", "apply_not_tuned", epochs=2)
        legacy_failures = [test for test in time_series_tester.failed_tests if 'LSTM apply_not_tuned' in test]
        # This is optional - legacy mode may not be supported
        if legacy_failures:
            pytest.skip("Legacy apply_not_tuned mode not supported (expected)")
    except Exception:
        pytest.skip("Legacy apply_not_tuned mode not supported (expected)")


# Performance and edge case tests
@pytest.mark.slow
def test_small_dataset_handling(time_series_tester):
    """Test handling of very small datasets."""
    # This would require creating a smaller test dataset
    # For now, just verify current test data works
    time_series_tester.test_data_processing()
    data_failures = [test for test in time_series_tester.failed_tests if 'Data' in test]
    assert len(data_failures) == 0, f"Small dataset handling failures: {data_failures}"


if __name__ == "__main__":
    # Run pytest when executed directly
    pytest.main([__file__, "-v"])