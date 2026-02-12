"""
Visualization Utilities for MS3SEG Dataset
Functions for plotting images, masks, predictions, and results
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import seaborn as sns
import pandas as pd


# Publication-ready matplotlib settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'png',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})


class MS3SEGVisualizer:
    """Visualization class for MS3SEG dataset"""
    
    def __init__(self, save_dir=None):
        """
        Initialize visualizer
        
        Args:
            save_dir: Directory to save figures
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Color scheme for tri-mask visualization
        self.colors = {
            0: (0, 0, 0),        # Background: Black
            1: (0, 0, 255),      # Ventricles: Blue
            2: (0, 255, 0),      # Normal WMH: Green
            3: (255, 0, 0)       # Abnormal WMH: Red
        }
        
        self.class_names = {
            0: 'Background',
            1: 'Ventricles',
            2: 'Normal WMH',
            3: 'Abnormal WMH'
        }
    
    def mask_to_rgb(self, mask):
        """
        Convert integer mask to RGB image
        
        Args:
            mask: 2D array with integer class labels
        
        Returns:
            RGB image (H, W, 3)
        """
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, color in self.colors.items():
            rgb[mask == class_id] = color
        return rgb
    
    def plot_sample_with_mask(self, image, mask, title=None, save_name=None):
        """
        Plot single image with overlay mask
        
        Args:
            image: 2D grayscale image
            mask: 2D integer mask
            title: Plot title
            save_name: Filename to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('FLAIR Image')
        axes[0].axis('off')
        
        # Mask
        mask_rgb = self.mask_to_rgb(mask)
        axes[1].imshow(mask_rgb)
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(image, cmap='gray')
        axes[2].imshow(mask_rgb, alpha=0.4)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        # Add legend
        patches = [mpatches.Patch(color=np.array(self.colors[i])/255, 
                                 label=self.class_names[i]) 
                  for i in range(1, 4)]  # Skip background
        fig.legend(handles=patches, loc='lower center', ncol=3, 
                  bbox_to_anchor=(0.5, -0.05))
        
        if title:
            fig.suptitle(title)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f"{save_name}.png")
            plt.close()
        else:
            plt.show()
    
    def plot_predictions_comparison(self, image, ground_truth, predictions_dict,
                                   scenario='multi_class', save_name=None):
        """
        Compare predictions from multiple models
        
        Args:
            image: Input FLAIR image
            ground_truth: Ground truth mask
            predictions_dict: Dictionary {model_name: prediction}
            scenario: Segmentation scenario
            save_name: Filename to save figure
        """
        n_models = len(predictions_dict)
        fig, axes = plt.subplots(2, n_models + 1, figsize=(4*(n_models+1), 8))
        
        # First column: input and ground truth
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Input FLAIR')
        axes[0, 0].axis('off')
        
        if scenario == 'multi_class':
            gt_rgb = self.mask_to_rgb(ground_truth)
            axes[1, 0].imshow(gt_rgb)
        else:
            axes[1, 0].imshow(ground_truth, cmap='gray')
        axes[1, 0].set_title('Ground Truth')
        axes[1, 0].axis('off')
        
        # Other columns: model predictions
        for idx, (model_name, prediction) in enumerate(predictions_dict.items(), 1):
            # Top row: overlay
            axes[0, idx].imshow(image, cmap='gray')
            if scenario == 'multi_class':
                pred_rgb = self.mask_to_rgb(prediction)
                axes[0, idx].imshow(pred_rgb, alpha=0.4)
            else:
                axes[0, idx].imshow(prediction, cmap='Reds', alpha=0.4)
            axes[0, idx].set_title(f'{model_name} Overlay')
            axes[0, idx].axis('off')
            
            # Bottom row: prediction only
            if scenario == 'multi_class':
                axes[1, idx].imshow(pred_rgb)
            else:
                axes[1, idx].imshow(prediction, cmap='gray')
            axes[1, idx].set_title(f'{model_name} Prediction')
            axes[1, idx].axis('off')
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f"{save_name}.png")
            plt.close()
        else:
            plt.show()
    
    def plot_error_map(self, ground_truth, prediction, save_name=None):
        """
        Visualize prediction errors
        
        Args:
            ground_truth: Ground truth binary mask
            prediction: Predicted binary mask
            save_name: Filename to save figure
        """
        # Create error map: TP=green, FP=red, FN=blue
        error_map = np.zeros((*ground_truth.shape, 3), dtype=np.uint8)
        
        tp = (ground_truth == 1) & (prediction == 1)
        fp = (ground_truth == 0) & (prediction == 1)
        fn = (ground_truth == 1) & (prediction == 0)
        
        error_map[tp] = [0, 255, 0]   # True Positive: Green
        error_map[fp] = [255, 0, 0]   # False Positive: Red
        error_map[fn] = [0, 0, 255]   # False Negative: Blue
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(ground_truth, cmap='gray')
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        
        axes[1].imshow(prediction, cmap='gray')
        axes[1].set_title('Prediction')
        axes[1].axis('off')
        
        axes[2].imshow(error_map)
        axes[2].set_title('Error Map')
        axes[2].axis('off')
        
        # Add legend
        patches = [
            mpatches.Patch(color='green', label='True Positive'),
            mpatches.Patch(color='red', label='False Positive'),
            mpatches.Patch(color='blue', label='False Negative')
        ]
        fig.legend(handles=patches, loc='lower center', ncol=3)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f"{save_name}.png")
            plt.close()
        else:
            plt.show()
    
    def plot_training_history(self, history, save_name='training_history'):
        """
        Plot training history (loss and metrics)
        
        Args:
            history: Keras history object or dict with 'loss', 'val_loss', etc.
            save_name: Filename to save figure
        """
        if hasattr(history, 'history'):
            history = history.history
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy/Dice (if available)
        metric_key = None
        for key in history.keys():
            if 'dice' in key.lower() or 'accuracy' in key.lower():
                metric_key = key
                break
        
        if metric_key:
            axes[1].plot(history[metric_key], label=f'Training {metric_key}')
            val_key = f'val_{metric_key}'
            if val_key in history:
                axes[1].plot(history[val_key], label=f'Validation {metric_key}')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel(metric_key)
            axes[1].set_title(f'Training and Validation {metric_key}')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f"{save_name}.png")
            plt.close()
        else:
            plt.show()
    
    def plot_kfold_results(self, results_df, save_name='kfold_results'):
        """
        Visualize k-fold cross-validation results
        
        Args:
            results_df: DataFrame with columns ['Model', 'Fold', 'Dice', 'IoU', 'HD95']
            save_name: Filename to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        metrics = ['Dice', 'IoU', 'HD95']
        
        for idx, metric in enumerate(metrics):
            if metric in results_df.columns:
                sns.boxplot(data=results_df, x='Model', y=metric, ax=axes[idx])
                axes[idx].set_title(f'{metric} Score Across Folds')
                axes[idx].set_xlabel('Model')
                axes[idx].set_ylabel(metric)
                axes[idx].tick_params(axis='x', rotation=45)
                axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f"{save_name}.png")
            plt.close()
        else:
            plt.show()
    
    def create_summary_table(self, results_df, save_name='results_summary'):
        """
        Create a summary table of results
        
        Args:
            results_df: DataFrame with model results
            save_name: Filename to save table
        """
        # Group by model and calculate statistics
        summary = results_df.groupby('Model').agg({
            'Dice': ['mean', 'std'],
            'IoU': ['mean', 'std'],
            'HD95': ['mean', 'std']
        }).round(4)
        
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(summary)
        print("="*80)
        
        if save_name and self.save_dir:
            summary.to_csv(self.save_dir / f"{save_name}.csv")
            print(f"\nSummary saved to: {self.save_dir / f'{save_name}.csv'}")
        
        return summary


if __name__ == "__main__":
    # Example usage
    visualizer = MS3SEGVisualizer(save_dir="figures")
    
    # Create dummy data
    image = np.random.rand(256, 256)
    mask = np.random.randint(0, 4, (256, 256))
    
    # Plot sample
    visualizer.plot_sample_with_mask(image, mask, 
                                    title="Example Visualization",
                                    save_name="example")
    
    print("Visualization example complete!")
