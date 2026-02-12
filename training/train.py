"""
Training Script for MS3SEG Dataset
Implements k-fold cross-validation with multiple models and scenarios
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
import argparse
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models import get_model
from utils import (
    MS3SEGDataLoader, calculate_class_weights,
    unified_focal_loss, dice_loss, multiclass_dice_loss,
    MS3SEGVisualizer
)


class MS3SEGTrainer:
    """Main training class for MS3SEG experiments"""
    
    def __init__(self, config_path='config.json'):
        """
        Initialize trainer with configuration
        
        Args:
            config_path: Path to configuration JSON file
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Setup paths
        self.setup_paths()
        
        # Initialize data loader
        self.data_loader = MS3SEGDataLoader(
            data_dir=self.config['data']['data_dir'],
            target_size=tuple(self.config['data']['target_size'])
        )
        
        # Initialize visualizer
        self.visualizer = MS3SEGVisualizer(
            save_dir=self.results_dir / 'figures'
        )
        
        # Set random seeds
        self.set_random_seeds()
        
        print(f"Trainer initialized. Results will be saved to: {self.results_dir}")
    
    def setup_paths(self):
        """Create directory structure for results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(self.config['output']['results_dir']) / f"experiment_{timestamp}"
        
        subdirs = ['models', 'figures', 'predictions', 'logs', 'tables']
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        for subdir in subdirs:
            (self.results_dir / subdir).mkdir(exist_ok=True)
        
        # Save config to results directory
        with open(self.results_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def set_random_seeds(self):
        """Set random seeds for reproducibility"""
        seed = self.config['cross_validation']['random_state']
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    def get_loss_function(self, scenario, class_weights=None):
        """
        Get loss function based on configuration
        
        Args:
            scenario: Segmentation scenario
            class_weights: Optional class weights
        
        Returns:
            Loss function
        """
        loss_name = self.config['training']['loss_function']
        num_classes = self.config['scenarios'][scenario]['num_classes']
        
        if loss_name == 'unified_focal_loss':
            return unified_focal_loss(class_weights=class_weights)
        elif loss_name == 'multiclass_dice':
            return multiclass_dice_loss(num_classes=num_classes, class_weights=class_weights)
        elif loss_name == 'dice':
            return dice_loss()
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")
    
    def compile_model(self, model, scenario, class_weights=None):
        """
        Compile model with optimizer and loss
        
        Args:
            model: Keras model
            scenario: Segmentation scenario
            class_weights: Optional class weights
        
        Returns:
            Compiled model
        """
        optimizer = keras.optimizers.Adam(
            learning_rate=self.config['training']['learning_rate']
        )
        
        loss = self.get_loss_function(scenario, class_weights)
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
        
        return model
    
    def get_callbacks(self, model_save_path):
        """
        Create training callbacks
        
        Args:
            model_save_path: Path to save best model
        
        Returns:
            List of callbacks
        """
        callbacks_list = [
            keras.callbacks.ModelCheckpoint(
                str(model_save_path),
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['training']['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config['training']['reduce_lr_factor'],
                patience=self.config['training']['reduce_lr_patience'],
                min_lr=self.config['training']['min_lr'],
                verbose=1
            ),
            keras.callbacks.CSVLogger(
                str(self.results_dir / 'logs' / 'training_log.csv')
            )
        ]
        
        return callbacks_list
    
    def train_fold(self, model_name, scenario, fold_idx, train_patients, val_patients):
        """
        Train a single fold
        
        Args:
            model_name: Name of the model
            scenario: Segmentation scenario
            fold_idx: Fold index
            train_patients: List of training patient IDs
            val_patients: List of validation patient IDs
        
        Returns:
            Trained model and history
        """
        print(f"\n{'='*80}")
        print(f"Training {model_name} - {scenario} - Fold {fold_idx}")
        print(f"Train patients: {len(train_patients)}, Val patients: {len(val_patients)}")
        print(f"{'='*80}\n")
        
        # Load data
        print("Loading training data...")
        train_images, train_masks = self.data_loader.load_dataset(train_patients, scenario)
        
        print("Loading validation data...")
        val_images, val_masks = self.data_loader.load_dataset(val_patients, scenario)
        
        # Convert masks to one-hot for multi-class scenarios
        num_classes = self.config['scenarios'][scenario]['num_classes']
        if num_classes > 2:
            train_masks = keras.utils.to_categorical(train_masks, num_classes)
            val_masks = keras.utils.to_categorical(val_masks, num_classes)
        else:
            # For binary, add channel dimension
            train_masks = np.expand_dims(train_masks, axis=-1)
            val_masks = np.expand_dims(val_masks, axis=-1)
        
        # Calculate class weights
        class_weights = None
        if num_classes > 2:
            flat_masks = train_masks.argmax(axis=-1)
            class_weights = calculate_class_weights(flat_masks, num_classes)
            print(f"Class weights: {class_weights}")
        
        # Build model
        print(f"Building {model_name} model...")
        input_shape = (*self.config['data']['target_size'], 1)
        model_params = self.config['models']['default_params'][model_name].copy()
        
        model = get_model(
            model_name,
            input_shape=input_shape,
            num_classes=num_classes,
            **model_params
        )
        
        # Compile model
        model = self.compile_model(model, scenario, class_weights)
        
        print(f"Total parameters: {model.count_params():,}")
        
        # Setup callbacks
        model_save_path = (self.results_dir / 'models' / scenario / 
                          f"{model_name}_fold{fold_idx}.h5")
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        
        callbacks = self.get_callbacks(model_save_path)
        
        # Train model
        print("\nStarting training...")
        history = model.fit(
            train_images, train_masks,
            validation_data=(val_images, val_masks),
            epochs=self.config['training']['epochs'],
            batch_size=self.config['training']['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Plot training history
        self.visualizer.plot_training_history(
            history,
            save_name=f"{scenario}_{model_name}_fold{fold_idx}_history"
        )
        
        return model, history
    
    def run_kfold_experiment(self, scenario='multi_class', models=None):
        """
        Run complete k-fold cross-validation experiment
        
        Args:
            scenario: Segmentation scenario to run
            models: List of model names to evaluate (None = all models)
        
        Returns:
            Dictionary with results
        """
        if models is None:
            models = self.config['models']['to_compare']
        
        # Load or create splits
        splits_file = self.config['cross_validation']['splits_file']
        if Path(splits_file).exists():
            print(f"Loading existing splits from: {splits_file}")
            splits = self.data_loader.load_splits(splits_file)
        else:
            print("Creating new k-fold splits...")
            all_patients = [f"{i:03d}" for i in range(1, 101)]
            splits = self.data_loader.create_kfold_splits(
                all_patients,
                n_splits=self.config['cross_validation']['n_folds'],
                test_size=self.config['cross_validation']['test_split'],
                random_state=self.config['cross_validation']['random_state'],
                save_path=splits_file
            )
        
        results = {
            'scenario': scenario,
            'models': {},
            'folds': []
        }
        
        # Train each model on each fold
        for model_name in models:
            results['models'][model_name] = {
                'fold_histories': [],
                'fold_models': []
            }
            
            for fold_data in splits['folds']:
                fold_idx = fold_data['fold']
                
                model, history = self.train_fold(
                    model_name,
                    scenario,
                    fold_idx,
                    fold_data['train_patients'],
                    fold_data['val_patients']
                )
                
                results['models'][model_name]['fold_histories'].append(history.history)
                results['models'][model_name]['fold_models'].append(model)
        
        # Save results
        results_file = self.results_dir / 'tables' / f'{scenario}_results.json'
        with open(results_file, 'w') as f:
            # Convert history objects to serializable format
            serializable_results = {
                'scenario': results['scenario'],
                'models': {
                    name: {'fold_histories': data['fold_histories']}
                    for name, data in results['models'].items()
                }
            }
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        return results


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train MS3SEG models')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file')
    parser.add_argument('--scenario', type=str, default='multi_class',
                       choices=['multi_class', 'three_class', 'binary_abnormal_wmh', 'binary_ventricles'],
                       help='Segmentation scenario')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Models to train (default: all models in config)')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = MS3SEGTrainer(config_path=args.config)
    
    # Run experiment
    results = trainer.run_kfold_experiment(
        scenario=args.scenario,
        models=args.models
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print(f"Results saved to: {trainer.results_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
