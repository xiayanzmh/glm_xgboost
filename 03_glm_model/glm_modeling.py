#!/usr/bin/env python3
"""
GLM Model Development and Tuning
GLM vs XGBoost Modeling Project

This script develops and tunes Generalized Linear Models (Logistic Regression) with:
1. Multiple regularization approaches (L1, L2, Elastic Net)
2. Hyperparameter tuning using GridSearchCV and RandomizedSearchCV
3. Cross-validation with different scoring metrics
4. Model evaluation and feature importance analysis
5. Handling class imbalance strategies
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score,
    StratifiedKFold, validation_curve
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, auc, f1_score,
    accuracy_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

def load_processed_data(data_dir):
    """Load preprocessed data"""
    print("="*60)
    print("LOADING PROCESSED DATA")
    print("="*60)

    data_path = Path(data_dir)

    # Load datasets
    datasets = {}
    required_files = [
        'X_train_scaled', 'X_val_scaled', 'X_test_scaled',
        'y_train', 'y_val', 'y_test',
        'X_train_scaled_smote', 'y_train_smote',
        'class_weights', 'scaler'
    ]

    for file_name in required_files:
        file_path = data_path / f"{file_name}.pkl"
        with open(file_path, 'rb') as f:
            datasets[file_name] = pickle.load(f)
        print(f"✓ Loaded {file_name}: {datasets[file_name].shape if hasattr(datasets[file_name], 'shape') else 'N/A'}")

    return datasets

def evaluate_model_performance(model, X, y, dataset_name=""):
    """Comprehensive model evaluation"""

    # Predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Basic metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc_score = roc_auc_score(y, y_pred_proba)

    print(f"\n{dataset_name} Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc_score:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def perform_hyperparameter_tuning(X_train, y_train, X_val, y_val, class_weights=None, cv_folds=5):
    """Comprehensive hyperparameter tuning for Logistic Regression"""
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)

    # Define parameter grids for different approaches
    param_grids = {
        'lasso': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000, 2000]
        },
        'ridge': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['liblinear', 'lbfgs', 'saga'],
            'max_iter': [1000, 2000]
        },
        'elastic_net': {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'penalty': ['elasticnet'],
            'solver': ['saga'],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'max_iter': [1000, 2000]
        }
    }

    # Setup cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    best_models = {}
    tuning_results = {}

    for reg_type, param_grid in param_grids.items():
        print(f"\nTuning {reg_type.upper()} regularization...")

        # Create base model
        base_model = LogisticRegression(
            random_state=42,
            class_weight=class_weights
        )

        # Grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='roc_auc',  # Primary metric for binary classification
            n_jobs=-1,
            verbose=1
        )

        # Fit grid search
        grid_search.fit(X_train, y_train)

        # Store best model
        best_models[reg_type] = grid_search.best_estimator_

        # Evaluate on validation set
        val_score = roc_auc_score(y_val, grid_search.best_estimator_.predict_proba(X_val)[:, 1])

        tuning_results[reg_type] = {
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'val_score': val_score,
            'model': grid_search.best_estimator_
        }

        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Best CV score: {grid_search.best_score_:.4f}")
        print(f"  Validation score: {val_score:.4f}")

    # Find overall best model
    best_reg_type = max(tuning_results.keys(),
                       key=lambda x: tuning_results[x]['val_score'])
    best_overall_model = tuning_results[best_reg_type]['model']

    print(f"\n🏆 Best regularization: {best_reg_type.upper()}")
    print(f"Best validation AUC: {tuning_results[best_reg_type]['val_score']:.4f}")

    return best_overall_model, tuning_results

def analyze_feature_importance(model, feature_names, top_n=20):
    """Analyze and visualize feature importance"""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)

    # Get coefficients
    coefficients = model.coef_[0]

    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    })

    # Sort by absolute coefficient
    feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)

    print(f"Top {top_n} most important features:")
    print(feature_importance.head(top_n))

    return feature_importance

def cross_validate_models(models_dict, X_train, y_train, cv_folds=5):
    """Perform cross-validation comparison of models"""
    print("\n" + "="*60)
    print("CROSS-VALIDATION COMPARISON")
    print("="*60)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    cv_results = {}

    for name, model in models_dict.items():
        print(f"\nCross-validating {name}...")

        # Multiple scoring metrics
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

        scores = {}
        for metric in scoring_metrics:
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric)
            scores[metric] = {
                'mean': cv_scores.mean(),
                'std': cv_scores.std(),
                'scores': cv_scores
            }
            print(f"  {metric}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        cv_results[name] = scores

    return cv_results

def create_visualizations(models_results, output_dir):
    """Create comprehensive visualizations"""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)

    plots_dir = Path(output_dir) / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8')

    # 1. Model comparison plot
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    datasets = ['train', 'validation', 'test']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Extract metric values for plotting
        train_vals = [models_results[model_name]['train'][metric] for model_name in models_results.keys()]
        val_vals = [models_results[model_name]['validation'][metric] for model_name in models_results.keys()]
        test_vals = [models_results[model_name]['test'][metric] for model_name in models_results.keys()]

        x = np.arange(len(models_results))
        width = 0.25

        ax.bar(x - width, train_vals, width, label='Train', alpha=0.8)
        ax.bar(x, val_vals, width, label='Validation', alpha=0.8)
        ax.bar(x + width, test_vals, width, label='Test', alpha=0.8)

        ax.set_xlabel('Models')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(list(models_results.keys()), rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Remove empty subplot
    if len(metrics) < len(axes):
        fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.savefig(plots_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✓ Visualizations saved in {plots_dir}/")

def train_glm_models(datasets):
    """Train multiple GLM variants"""
    print("\n" + "="*60)
    print("TRAINING GLM MODELS")
    print("="*60)

    X_train_scaled = datasets['X_train_scaled']
    X_val_scaled = datasets['X_val_scaled']
    X_test_scaled = datasets['X_test_scaled']
    y_train = datasets['y_train']
    y_val = datasets['y_val']
    y_test = datasets['y_test']

    X_train_smote = datasets['X_train_scaled_smote']
    y_train_smote = datasets['y_train_smote']
    class_weights = datasets['class_weights']

    # Feature names
    feature_names = X_train_scaled.columns.tolist()

    models = {}
    results = {}

    # 1. Basic Logistic Regression (no class imbalance handling)
    print("\n1. Training Basic Logistic Regression...")
    lr_basic = LogisticRegression(random_state=42, max_iter=1000)
    lr_basic.fit(X_train_scaled, y_train)
    models['Basic_LR'] = lr_basic

    # 2. Logistic Regression with Class Weights
    print("\n2. Training Logistic Regression with Class Weights...")
    lr_weighted = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight=class_weights
    )
    lr_weighted.fit(X_train_scaled, y_train)
    models['Weighted_LR'] = lr_weighted

    # 3. Logistic Regression with SMOTE
    print("\n3. Training Logistic Regression with SMOTE data...")
    lr_smote = LogisticRegression(random_state=42, max_iter=1000)
    lr_smote.fit(X_train_smote, y_train_smote)
    models['SMOTE_LR'] = lr_smote

    # 4. Hyperparameter-tuned models
    print("\n4. Hyperparameter Tuning...")

    # Tune on weighted data
    best_weighted_model, tuning_results_weighted = perform_hyperparameter_tuning(
        X_train_scaled, y_train, X_val_scaled, y_val, class_weights=class_weights
    )
    models['Best_Weighted_LR'] = best_weighted_model

    # Tune on SMOTE data
    best_smote_model, tuning_results_smote = perform_hyperparameter_tuning(
        X_train_smote, y_train_smote, X_val_scaled, y_val, class_weights=None
    )
    models['Best_SMOTE_LR'] = best_smote_model

    # Evaluate all models
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)

    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}:")
        results[model_name] = {}

        # Use appropriate training data for evaluation
        if 'SMOTE' in model_name:
            train_X, train_y = X_train_smote, y_train_smote
        else:
            train_X, train_y = X_train_scaled, y_train

        results[model_name]['train'] = evaluate_model_performance(
            model, train_X, train_y, "Training"
        )
        results[model_name]['validation'] = evaluate_model_performance(
            model, X_val_scaled, y_val, "Validation"
        )
        results[model_name]['test'] = evaluate_model_performance(
            model, X_test_scaled, y_test, "Test"
        )

    # Cross-validation comparison
    cv_results = cross_validate_models(models, X_train_scaled, y_train)

    # Feature importance analysis for best model
    best_model_name = max(models.keys(),
                         key=lambda x: results[x]['validation']['auc'])
    best_model = models[best_model_name]

    print(f"\n🏆 Best model: {best_model_name}")
    print(f"Best validation AUC: {results[best_model_name]['validation']['auc']:.4f}")

    feature_importance = analyze_feature_importance(best_model, feature_names)

    return models, results, cv_results, feature_importance, best_model_name

def save_glm_results(models, results, cv_results, feature_importance, best_model_name, output_dir):
    """Save all GLM results"""
    print("\n" + "="*60)
    print("SAVING GLM RESULTS")
    print("="*60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save models
    models_path = output_path / 'models'
    models_path.mkdir(exist_ok=True)

    for model_name, model in models.items():
        model_file = models_path / f"{model_name}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Saved model: {model_file}")

    # Save results
    results_file = output_path / 'evaluation_results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"✓ Saved evaluation results: {results_file}")

    # Save CV results
    cv_file = output_path / 'cv_results.pkl'
    with open(cv_file, 'wb') as f:
        pickle.dump(cv_results, f)
    print(f"✓ Saved CV results: {cv_file}")

    # Save feature importance
    feature_importance_file = output_path / 'feature_importance.csv'
    feature_importance.to_csv(feature_importance_file, index=False)
    print(f"✓ Saved feature importance: {feature_importance_file}")

    # Save best model info
    best_model_info = {
        'best_model_name': best_model_name,
        'best_model': models[best_model_name],
        'performance': results[best_model_name]
    }

    best_model_file = output_path / 'best_glm_model.pkl'
    with open(best_model_file, 'wb') as f:
        pickle.dump(best_model_info, f)
    print(f"✓ Saved best model: {best_model_file}")

    print(f"\nAll GLM results saved to: {output_path}")

def main():
    """Main GLM modeling pipeline"""
    print("STARTING GLM MODEL DEVELOPMENT")
    print("="*80)

    # Configuration
    DATA_DIR = '02_preprocessing/processed_data'
    OUTPUT_DIR = '03_glm_model/results'

    # Load data
    datasets = load_processed_data(DATA_DIR)

    # Train and evaluate models
    models, results, cv_results, feature_importance, best_model_name = train_glm_models(datasets)

    # Create visualizations
    create_visualizations(results, OUTPUT_DIR)

    # Save results
    save_glm_results(models, results, cv_results, feature_importance, best_model_name, OUTPUT_DIR)

    # Summary
    print("\n" + "="*60)
    print("GLM MODELING SUMMARY")
    print("="*60)
    print(f"✓ Models trained: {len(models)}")
    print(f"✓ Best model: {best_model_name}")
    print(f"✓ Best validation AUC: {results[best_model_name]['validation']['auc']:.4f}")
    print(f"✓ Results saved to: {OUTPUT_DIR}")

    return models, results, best_model_name

if __name__ == "__main__":
    models, results, best_model_name = main()