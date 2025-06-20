"""
Visualization Utilities
Handles plotting and visualization functions for training results.

Author: Soroosh Shalileh
Email: sr.shalileh@gmail.com
Year: 2025
"""

import matplotlib.pyplot as plt
import os
from pathlib import Path
from typing import Dict, List
from .file_utils import create_directory_safely


def save_training_plots(history: Dict[str, List[float]], save_dir: Path, model_name: str):
    """
    Save training and validation loss/metrics plots.
    
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
        # Plot training and validation loss
        plt.figure(figsize=(10, 6))
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
        plt.figure(figsize=(10, 6))
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
        plt.figure(figsize=(10, 6))
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
        
        print(f"Successfully saved training plots to: {save_dir}")
        
    except Exception as e:
        print(f"Error saving training plots: {e}") 