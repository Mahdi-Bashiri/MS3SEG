"""
Evaluation Script for MS3SEG Dataset
Evaluate trained models and generate comprehensive reports
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
import pandas as pd
import argparse
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils import (
    MS3SEGDataLoader,
    calculate_multiclass_metrics,
    calculate_binary_metrics,
    MS3SEGVisualizer
)


class MS3SEGEvaluator:
    """Evaluation class for trained models"""
    
    def __init__(self, results_dir, config_path='config.json'):
        """
        Initialize evaluator
        
        Args:
            results_dir: Directory containing trained models and results
            config_path: Path to configuration file
        """
        self.results_dir = Path(results_dir)
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize data loader
        self.data_loader = MS3SEGDataLoader(
            data_dir=self.config['data']['data_dir'],
            target_size=tuple(self.config['data']['target_size'])
        )
        
        # Initialize visualizer
        self.visualizer = MS3SEGVisualizer(
            save_dir=self.results_dir / 'figures'
        )
    
    def load_model(self, model_path):
        """
        Load a trained model
        
        Args:
            model_path: Path to saved model
        
        Returns:
            Loaded Keras model
        """
        return keras.models.load_model(model_path, compile=False)
    
    def evaluate_fold(self, model_path, test_patients, scenario):
        """
        Evaluate model on test patients
        
        Args:
            model_path: Path to saved model
            test_patients: List of test patient IDs
            scenario: Segmentation scenario
        
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating: {model_path.name}")
        
        # Load model
        model = self.load_model(model_path)
        
        # Load test data
        test_images, test_masks = self.data_loader.load_dataset(test_patients, scenario)
        
        # Make predictions
        print("Generating predictions...")
        predictions = model.predict(test_images, batch_size=4, verbose=1)
        
        # Convert predictions to class labels
        num_classes = self.config['scenarios'][scenario]['num_classes']
        if num_classes > 2:
            pred_classes = np.argmax(predictions, axis=-1)
        else:
            pred_classes = (predictions.squeeze() > 0.5).astype(np.uint8)
        
        # Calculate metrics
        print("Calculating metrics...")
        if num_classes > 2:
            metrics = calculate_multiclass_metrics(
                test_masks,
                pred_classes,
                num_classes,
                class_names=self.config['scenarios'][scenario]['class_names']
            )
        else:
            metrics = calculate_binary_metrics(test_masks, pred_classes)
        
        return {
            'metrics': metrics,
            'predictions': pred_classes,
            'ground_truth': test_masks
        }
    
    def evaluate_scenario(self, scenario, test_patients=None):
        """
        Evaluate all models for a given scenario
        
        Args:
            scenario: Segmentation scenario
            test_patients: List of test patient IDs (if None, loads from splits)
        
        Returns:
            DataFrame with all results
        """
        print(f"\n{'='*80}")
        print(f"Evaluating scenario: {scenario}")
        print(f"{'='*80}\n")
        
        # Load test patients if not provided
        if test_patients is None:
            splits_file = self.config['cross_validation']['splits_file']
            with open(splits_file, 'r') as f:
                splits = json.load(f)
            test_patients = splits['test_patients']
        
        print(f"Test patients: {len(test_patients)}")
        
        # Find all model files for this scenario
        models_dir = self.results_dir / 'models' / scenario
        if not models_dir.exists():
            print(f"No models found for scenario: {scenario}")
            return None
        
        model_files = sorted(models_dir.glob('*.h5'))
        
        if not model_files:
            print(f"No model files found in: {models_dir}")
            return None
        
        print(f"Found {len(model_files)} models to evaluate\n")
        
        # Evaluate each model
        all_results = []
        
        for model_path in model_files:
            # Extract model name and fold from filename
            parts = model_path.stem.split('_fold')
            model_name = parts[0]
            fold_idx = int(parts[1]) if len(parts) > 1 else 0
            
            try:
                results = self.evaluate_fold(model_path, test_patients, scenario)
                
                # Extract metrics
                metrics = results['metrics']
                
                if 'overall' in metrics:
                    # Multi-class results
                    row = {
                        'Model': model_name,
                        'Fold': fold_idx,
                        'Mean_Dice': metrics['overall']['Mean_Dice'],
                        'Mean_IoU': metrics['overall']['Mean_IoU'],
                        'Mean_HD95': metrics['overall']['Mean_HD95']
                    }
                    
                    # Add per-class metrics
                    for class_name, class_metrics in metrics['per_class'].items():
                        row[f'{class_name}_Dice'] = class_metrics['Dice']
                        row[f'{class_name}_IoU'] = class_metrics['IoU']
                        row[f'{class_name}_HD95'] = class_metrics['HD95']
                else:
                    # Binary results
                    row = {
                        'Model': model_name,
                        'Fold': fold_idx,
                        'Dice': metrics['Dice'],
                        'IoU': metrics['IoU'],
                        'HD95': metrics['HD95']
                    }
                
                all_results.append(row)
                
                print(f"✓ {model_name} (Fold {fold_idx}) evaluated successfully")
                
            except Exception as e:
                print(f"✗ Error evaluating {model_path.name}: {e}")
                continue
        
        # Create DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Save results
        csv_path = self.results_dir / 'tables' / f'{scenario}_evaluation.csv'
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        self.visualizer.plot_kfold_results(results_df, save_name=f'{scenario}_kfold_comparison')
        self.visualizer.create_summary_table(results_df, save_name=f'{scenario}_summary')
        
        return results_df
    
    def evaluate_all_scenarios(self):
        """Evaluate all scenarios"""
        results = {}
        
        for scenario in self.config['scenarios'].keys():
            print(f"\n{'#'*80}")
            print(f"# SCENARIO: {scenario}")
            print(f"{'#'*80}")
            
            results[scenario] = self.evaluate_scenario(scenario)
        
        return results
    
    def generate_sample_visualizations(self, scenario, model_name, num_samples=5):
        """
        Generate sample prediction visualizations
        
        Args:
            scenario: Segmentation scenario
            model_name: Name of the model
            num_samples: Number of samples to visualize
        """
        print(f"\nGenerating sample visualizations for {model_name} - {scenario}...")
        
        # Find model file
        models_dir = self.results_dir / 'models' / scenario
        model_files = list(models_dir.glob(f'{model_name}_fold*.h5'))
        
        if not model_files:
            print(f"No models found for {model_name} in {scenario}")
            return
        
        # Use first fold model
        model_path = model_files[0]
        model = self.load_model(model_path)
        
        # Load test data
        splits_file = self.config['cross_validation']['splits_file']
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        
        test_patients = splits['test_patients'][:num_samples]
        test_images, test_masks = self.data_loader.load_dataset(test_patients, scenario)
        
        # Generate predictions
        predictions = model.predict(test_images, batch_size=4)
        
        # Convert to class labels
        num_classes = self.config['scenarios'][scenario]['num_classes']
        if num_classes > 2:
            pred_classes = np.argmax(predictions, axis=-1)
        else:
            pred_classes = (predictions.squeeze() > 0.5).astype(np.uint8)
        
        # Visualize samples
        for i in range(min(num_samples, len(test_images))):
            if num_classes > 2:
                self.visualizer.plot_sample_with_mask(
                    test_images[i].squeeze(),
                    test_masks[i],
                    title=f'{model_name} - Sample {i+1}',
                    save_name=f'{scenario}_{model_name}_sample_{i+1}_gt'
                )
                
                self.visualizer.plot_sample_with_mask(
                    test_images[i].squeeze(),
                    pred_classes[i],
                    title=f'{model_name} Prediction - Sample {i+1}',
                    save_name=f'{scenario}_{model_name}_sample_{i+1}_pred'
                )
            else:
                self.visualizer.plot_error_map(
                    test_masks[i],
                    pred_classes[i],
                    save_name=f'{scenario}_{model_name}_sample_{i+1}_errors'
                )
        
        print(f"Sample visualizations saved to: {self.visualizer.save_dir}")


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate MS3SEG models')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing trained models')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file')
    parser.add_argument('--scenario', type=str, default=None,
                       help='Specific scenario to evaluate (default: all)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate sample visualizations')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = MS3SEGEvaluator(
        results_dir=args.results_dir,
        config_path=args.config
    )
    
    # Run evaluation
    if args.scenario:
        results = evaluator.evaluate_scenario(args.scenario)
        
        if args.visualize and results is not None:
            # Get unique models from results
            for model_name in results['Model'].unique():
                evaluator.generate_sample_visualizations(args.scenario, model_name)
    else:
        results = evaluator.evaluate_all_scenarios()
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print(f"Results saved to: {evaluator.results_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
