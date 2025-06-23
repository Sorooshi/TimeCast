"""
Time Series Forecasting Package
Main script for training and evaluating time series forecasting models.

Author: Soroosh Shalileh
Email: sr.shalileh@gmail.com
Year: 2025
"""

import argparse
from utils.workflow_manager import (
    load_model_class, setup_logging, run_tune_mode, 
    run_apply_mode, run_report_mode, get_mode_description
)
from utils.data_utils import get_data_path, load_and_validate_data, prepare_data_loaders
from utils.data_preprocessing import prepare_data_for_model


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(description='Time Series Forecasting with PyTorch')
    
    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                      help='Name of the model to use')
    
    parser.add_argument('--data_name', type=str, required=True,
                      help='Name of the dataset to use (without .csv extension)')
    
    # Optional arguments
    parser.add_argument('--data_path', type=str,
                      help='Full path to the data file. If not provided, '
                      'will look for {data_name}.csv in data/ directory')
    
    parser.add_argument('--mode', type=str, 
                      choices=['tune', 'apply', 'apply_not_tuned', 'report'],
                      default='apply', help='Mode to run the model in')
    
    parser.add_argument('--n_trials', type=int, default=100,
                      help='Number of trials for hyperparameter tuning')
    
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    
    parser.add_argument('--patience', type=int, default=25,
                      help='Patience for early stopping')
    
    parser.add_argument('--sequence_length', type=int, default=10,
                      help='Length of input sequences')
    
    parser.add_argument('--experiment_description', type=str, default=None,
                      help='Description of the experiment. If not provided, defaults to sequence length')
    
    # Data split arguments for small datasets
    parser.add_argument('--train_ratio', type=float, default=0.7,
                      help='Proportion of data to use for training (default: 0.7)')
    
    parser.add_argument('--val_ratio', type=float, default=0.1,
                      help='Proportion of data to use for validation (default: 0.1)')
    
    parser.add_argument('--normalization', type=str, default='minmax',
                      choices=['minmax', 'standard', 'none'],
                      help='Normalization method: minmax (0-1), standard (z-score), or none (default: minmax)')
    
    return parser


def print_mode_info(mode: str):
    """Print information about the selected mode."""
    print(f"\nMode: {mode}")
    print(f"Description: {get_mode_description(mode)}")
    print("-" * 50)


def main():
    """Main entry point for the application."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Print mode information
    print_mode_info(args.mode)
    
    # Handle report mode early (doesn't need data loading or model setup)
    if args.mode == 'report':
        run_report_mode(args)
        return
    
    try:
        # Load model class
        model_class, model_name = load_model_class(args.model)
        print(f"Loaded model: {model_name}")
        
        # Set up logging
        setup_logging(model_name, args.mode)
        
        # Load and validate data
        data_path = get_data_path(args.data_name, args.data_path)
        data, dates = load_and_validate_data(data_path)
        
        # Prepare data
        train_loader, val_loader, test_loader, input_size = prepare_data_for_model(
            data=data,
            dates=dates,
            sequence_length=args.sequence_length,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            normalization=args.normalization if args.normalization != 'none' else None
        )
        
        # Execute the appropriate workflow based on mode
        if args.mode == 'tune':
            run_tune_mode(model_class, model_name, train_loader, val_loader, 
                         test_loader, input_size, args)
        
        elif args.mode == 'apply':
            run_apply_mode(model_class, model_name, train_loader, val_loader, 
                          test_loader, input_size, args, use_tuned=True)
        
        elif args.mode == 'apply_not_tuned':
            run_apply_mode(model_class, model_name, train_loader, val_loader, 
                          test_loader, input_size, args, use_tuned=False)
        
        print(f"\nExperiment completed successfully!")
        print(f"Results saved for model: {model_name}")
        
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        raise


if __name__ == '__main__':
    main() 