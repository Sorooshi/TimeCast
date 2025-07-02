"""
Visualization Utilities
Handles plotting and visualization functions for training results.

Author: Soroosh Shalileh
Email: sr.shalileh@gmail.com
Year: 2025
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import Dict, List
from .file_utils import create_directory_safely


def save_training_plots(history: Dict[str, List[float]], save_dir: Path, model_name: str):
    """
    Save training and validation loss/metrics plots.
    Handles both individual fold plots and mean CV plots with std shading.
    
    Args:
        history: Dictionary containing training history
        save_dir: Directory to save the plots
        model_name: Name of the model for plot titles
    """
    plt.style.use('default')  
    
    # Create plots directory if it doesn't exist with robust creation
    if not create_directory_safely(save_dir):
        print(f"Error: Could not create plots directory: {save_dir}")
        return
    
    try:
        # Check if we have CV aggregated data (from K-fold)
        has_cv_data = any(key.startswith('cv_mean_') for key in history.keys())
        
        # Plot training and validation loss
        plt.figure(figsize=(12, 8))
        
        if has_cv_data:
            # Create subplot with two plots: best fold + CV mean
            plt.subplot(2, 1, 1)
            plt.plot(history['train_loss'], label='Training Loss (Best Fold)', alpha=0.8)
            plt.plot(history['val_loss'], label='Validation Loss (Best Fold)', alpha=0.8)
            plt.title(f'{model_name} - Best Fold Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 1, 2)
            epochs = range(len(history['cv_mean_train_loss']))
            mean_train = np.array(history['cv_mean_train_loss'])
            std_train = np.array(history['cv_std_train_loss'])
            mean_val = np.array(history['cv_mean_val_loss'])
            std_val = np.array(history['cv_std_val_loss'])
            
            plt.plot(epochs, mean_train, label='Training Loss (CV Mean)', color='blue')
            plt.fill_between(epochs, mean_train - std_train, mean_train + std_train, alpha=0.3, color='blue')
            plt.plot(epochs, mean_val, label='Validation Loss (CV Mean)', color='orange')
            plt.fill_between(epochs, mean_val - std_val, mean_val + std_val, alpha=0.3, color='orange')
            plt.title(f'{model_name} - K-Fold CV Mean Loss (±1 Std)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
        else:
            # Single plot for regular training
            plt.plot(history['train_loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title(f'{model_name} - Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'loss_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot R² Score
        plt.figure(figsize=(12, 8))
        
        if has_cv_data:
            plt.subplot(2, 1, 1)
            plt.plot(history['train_r2'], label='Training R² (Best Fold)', alpha=0.8)
            plt.plot(history['val_r2'], label='Validation R² (Best Fold)', alpha=0.8)
            plt.title(f'{model_name} - Best Fold Training and Validation R² Score')
            plt.xlabel('Epoch')
            plt.ylabel('R² Score')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 1, 2)
            epochs = range(len(history['cv_mean_train_r2']))
            mean_train = np.array(history['cv_mean_train_r2'])
            std_train = np.array(history['cv_std_train_r2'])
            mean_val = np.array(history['cv_mean_val_r2'])
            std_val = np.array(history['cv_std_val_r2'])
            
            plt.plot(epochs, mean_train, label='Training R² (CV Mean)', color='blue')
            plt.fill_between(epochs, mean_train - std_train, mean_train + std_train, alpha=0.3, color='blue')
            plt.plot(epochs, mean_val, label='Validation R² (CV Mean)', color='orange')
            plt.fill_between(epochs, mean_val - std_val, mean_val + std_val, alpha=0.3, color='orange')
            plt.title(f'{model_name} - K-Fold CV Mean R² Score (±1 Std)')
            plt.xlabel('Epoch')
            plt.ylabel('R² Score')
            plt.legend()
            plt.grid(True)
        else:
            plt.plot(history['train_r2'], label='Training R²')
            plt.plot(history['val_r2'], label='Validation R²')
            plt.title(f'{model_name} - Training and Validation R² Score')
            plt.xlabel('Epoch')
            plt.ylabel('R² Score')
            plt.legend()
            plt.grid(True)
            
        plt.tight_layout()
        plt.savefig(save_dir / 'r2_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot MAPE
        plt.figure(figsize=(12, 8))
        
        if has_cv_data:
            plt.subplot(2, 1, 1)
            plt.plot(history['train_mape'], label='Training MAPE (Best Fold)', alpha=0.8)
            plt.plot(history['val_mape'], label='Validation MAPE (Best Fold)', alpha=0.8)
            plt.title(f'{model_name} - Best Fold Training and Validation MAPE')
            plt.xlabel('Epoch')
            plt.ylabel('MAPE (%)')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 1, 2)
            epochs = range(len(history['cv_mean_train_mape']))
            mean_train = np.array(history['cv_mean_train_mape'])
            std_train = np.array(history['cv_std_train_mape'])
            mean_val = np.array(history['cv_mean_val_mape'])
            std_val = np.array(history['cv_std_val_mape'])
            
            plt.plot(epochs, mean_train, label='Training MAPE (CV Mean)', color='blue')
            plt.fill_between(epochs, mean_train - std_train, mean_train + std_train, alpha=0.3, color='blue')
            plt.plot(epochs, mean_val, label='Validation MAPE (CV Mean)', color='orange')
            plt.fill_between(epochs, mean_val - std_val, mean_val + std_val, alpha=0.3, color='orange')
            plt.title(f'{model_name} - K-Fold CV Mean MAPE (±1 Std)')
            plt.xlabel('Epoch')
            plt.ylabel('MAPE (%)')
            plt.legend()
            plt.grid(True)
        else:
            plt.plot(history['train_mape'], label='Training MAPE')
            plt.plot(history['val_mape'], label='Validation MAPE')
            plt.title(f'{model_name} - Training and Validation MAPE')
            plt.xlabel('Epoch')
            plt.ylabel('MAPE (%)')
            plt.legend()
            plt.grid(True)
            
        plt.tight_layout()
        plt.savefig(save_dir / 'mape_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_type = "with K-Fold CV mean and std shading" if has_cv_data else "single training run"
        print(f"Successfully saved training plots ({plot_type}) to: {save_dir}")
        
    except Exception as e:
        print(f"Error saving training plots: {e}") 