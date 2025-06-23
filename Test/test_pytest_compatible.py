#!/usr/bin/env python3
"""
Pytest-compatible test file for Time Series Forecasting Package
Wraps the existing TimeSeriesTester functionality for pytest execution.

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


@pytest.mark.slow
def test_lstm_apply_not_tuned(time_series_tester):
    """Test LSTM in apply_not_tuned mode."""
    time_series_tester.test_mode("LSTM", "apply_not_tuned", epochs=2)
    lstm_failures = [test for test in time_series_tester.failed_tests if 'LSTM apply_not_tuned' in test]
    assert len(lstm_failures) == 0, f"LSTM apply_not_tuned failures: {lstm_failures}"


@pytest.mark.slow
def test_mlp_apply_not_tuned(time_series_tester):
    """Test MLP in apply_not_tuned mode."""
    time_series_tester.test_mode("MLP", "apply_not_tuned", epochs=2)
    mlp_failures = [test for test in time_series_tester.failed_tests if 'MLP apply_not_tuned' in test]
    assert len(mlp_failures) == 0, f"MLP apply_not_tuned failures: {mlp_failures}"


@pytest.mark.slow
def test_tcn_apply_not_tuned(time_series_tester):
    """Test TCN in apply_not_tuned mode."""
    time_series_tester.test_mode("TCN", "apply_not_tuned", epochs=2)
    tcn_failures = [test for test in time_series_tester.failed_tests if 'TCN apply_not_tuned' in test]
    assert len(tcn_failures) == 0, f"TCN apply_not_tuned failures: {tcn_failures}"


def test_report_mode(time_series_tester):
    """Test report mode functionality."""
    time_series_tester.test_report_mode()
    report_failures = [test for test in time_series_tester.failed_tests if 'Report Mode' in test]
    assert len(report_failures) == 0, f"Report mode failures: {report_failures}"


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
def test_smoke_lstm(time_series_tester):
    """Quick smoke test for LSTM."""
    time_series_tester.test_mode("LSTM", "apply_not_tuned", epochs=1)
    lstm_failures = [test for test in time_series_tester.failed_tests if 'LSTM apply_not_tuned' in test]
    assert len(lstm_failures) == 0, f"LSTM smoke test failures: {lstm_failures}"


def test_smoke_mlp(time_series_tester):
    """Quick smoke test for MLP.""" 
    time_series_tester.test_mode("MLP", "apply_not_tuned", epochs=1)
    mlp_failures = [test for test in time_series_tester.failed_tests if 'MLP apply_not_tuned' in test]
    assert len(mlp_failures) == 0, f"MLP smoke test failures: {mlp_failures}"


if __name__ == "__main__":
    # Run pytest when executed directly
    pytest.main([__file__, "-v"]) 