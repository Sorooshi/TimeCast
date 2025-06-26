#!/usr/bin/env python3
"""
Test script for the new four modes of operation.
This script demonstrates how to use the new modes.

Author: Soroosh Shalileh
Email: sr.shalileh@gmail.com
Year: 2025
"""

import subprocess
import sys


def run_command(cmd):
    """Run a command and show its output."""
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def main():
    """Demonstrate the four modes of operation."""
    
    # Common parameters
    model = "LSTM"
    data_name = "merchant_synthetic"
    sequence_length = 10
    experiment_desc = "test_experiment"
    
    print("Testing the Four Modes of Operation")
    print("====================================")
    
    # Mode 1: Tune hyperparameters
    print("\n1. TUNE MODE - Optimize hyperparameters")
    tune_cmd = [
        "python", "main.py",
        "--model", model,
        "--data_name", data_name,
        "--mode", "tune",
        "--sequence_length", str(sequence_length),
        "--experiment_description", experiment_desc,
        "--n_trials", "5",  # Small number for testing
        "--epochs", "10",   # Small number for testing
        "--patience", "5"
    ]
    
    if not run_command(tune_cmd):
        print("Tune mode failed!")
        return False
    
    # Mode 2: Train with tuned parameters using K-fold CV
    print("\n2. TRAIN MODE - Train with tuned parameters using K-fold CV")
    train_cmd = [
        "python", "main.py",
        "--model", model,
        "--data_name", data_name,
        "--mode", "train",
        "--sequence_length", str(sequence_length),
        "--experiment_description", experiment_desc,
        "--k_folds", "3",   # Small number for testing
        "--epochs", "10",   # Small number for testing
        "--patience", "5"
    ]
    
    if not run_command(train_cmd):
        print("Train mode failed!")
        return False
    
    # Mode 3: Train with default parameters
    print("\n3. TRAIN MODE (DEFAULT) - Train with default parameters")
    train_default_cmd = [
        "python", "main.py",
        "--model", model,
        "--data_name", data_name,
        "--mode", "train",
        "--train_tuned", "false",
        "--sequence_length", str(sequence_length),
        "--experiment_description", experiment_desc,
        "--epochs", "10",   # Small number for testing
        "--patience", "5"
    ]
    
    if not run_command(train_default_cmd):
        print("Train default mode failed!")
        return False
    
    # Mode 4a: Predict with tuned model
    print("\n4a. PREDICT MODE - Using tuned model")
    predict_tuned_cmd = [
        "python", "main.py",
        "--model", model,
        "--data_name", data_name,
        "--mode", "predict",
        "--sequence_length", str(sequence_length),
        "--experiment_description", experiment_desc,
        "--predict_tuned", "true"
    ]
    
    if not run_command(predict_tuned_cmd):
        print("Predict mode (tuned) failed!")
        return False
    
    # Mode 4b: Predict with default model
    print("\n4b. PREDICT MODE - Using default model")
    predict_default_cmd = [
        "python", "main.py",
        "--model", model,
        "--data_name", data_name,
        "--mode", "predict",
        "--sequence_length", str(sequence_length),
        "--experiment_description", experiment_desc,
        "--predict_tuned", "false"
    ]
    
    if not run_command(predict_default_cmd):
        print("Predict mode (default) failed!")
        return False
    
    print("\n" + "="*80)
    print("ALL MODES COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 