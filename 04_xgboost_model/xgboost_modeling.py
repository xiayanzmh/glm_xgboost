#!/usr/bin/env python3
"""
XGBoost Model Development and Tuning
GLM vs XGBoost Modeling Project

This script develops and tunes XGBoost models with:
1. Hyperparameter optimization using Optuna for efficiency
2. Multiple evaluation metrics and early stopping
3. Feature importance analysis
4. Cross-validation strategies
5. Class imbalance handling strategies
6. Advanced XGBoost-specific techniques
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

import optuna
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_processed_data(data_dir):
    """Load preprocessed data for XGBoost"""
    print("="*60)
    print("LOADING PROCESSED DATA FOR XGBOOST")
    print("="*60)

    data_path = Path(data_dir)

    # Load datasets - XGBoost doesn't require scaled data
    datasets = {}
    required_files = [
        'X_train', 'X_val', 'X_test',  # Unscaled for XGBoost
        'y_train', 'y_val', 'y_test',
        'X_train_smote', 'y_train_smote',
        'class_weights'
    ]

    for file_name in required_files:
        file_path = data_path / f"{file_name}.pkl"
        with open(file_path, 'rb') as f:
            datasets[file_name] = pickle.load(f)
        print(f"✓ Loaded {file_name}: {datasets[file_name].shape if hasattr(datasets[file_name], 'shape') else 'N/A'}")

    return datasets

def evaluate_xgboost_performance(model, X, y, dataset_name=""):
    """Comprehensive XGBoost model evaluation"""

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

def objective_function(trial, X_train, y_train, X_val, y_val, sample_weight=None):
    """Optuna objective function for hyperparameter optimization"""

    # Hyperparameter suggestions
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'random_state': 42,
        'eval_metric': 'auc',
        'objective': 'binary:logistic'
    }

    # Create and train model
    model = xgb.XGBClassifier(**params)

    # Prepare evaluation set for early stopping
    eval_set = [(X_val, y_val)]

    # Fit with early stopping
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=eval_set,
        verbose=False
    )

    # Predict and calculate AUC
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, y_pred_proba)

    return auc_score

def perform_optuna_optimization(X_train, y_train, X_val, y_val, sample_weight=None, n_trials=100):
    """Perform hyperparameter optimization using Optuna"""
    print("\n" + "="*60)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("="*60)

    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # Optimize
    print(f"Starting optimization with {n_trials} trials...")
    study.optimize(
        lambda trial: objective_function(trial, X_train, y_train, X_val, y_val, sample_weight),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print(f"\nOptimization completed!")
    print(f"Best AUC: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")

    # Create best model
    best_params = study.best_params.copy()
    best_model = xgb.XGBClassifier(**best_params)

    # Train best model with early stopping
    eval_set = [(X_val, y_val)]
    best_model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=eval_set,
        verbose=False
    )

    return best_model, study.best_params, study

def analyze_xgboost_feature_importance(model, feature_names, top_n=20):
    """Analyze XGBoost feature importance"""
    print("\n" + "="*60)
    print("XGBOOST FEATURE IMPORTANCE ANALYSIS")
    print("="*60)

    # Get feature importances (multiple types)
    importance_types = ['weight', 'gain', 'cover']
    importance_results = {}

    for importance_type in importance_types:
        importance_scores = model.get_booster().get_score(importance_type=importance_type)

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': list(importance_scores.keys()),
            f'{importance_type}_importance': list(importance_scores.values())
        })

        # Sort by importance
        importance_df = importance_df.sort_values(f'{importance_type}_importance', ascending=False)
        importance_results[importance_type] = importance_df

        print(f"\nTop {top_n} features by {importance_type}:")
        print(importance_df.head(top_n))

    return importance_results

def cross_validate_xgboost(models_dict, X_train, y_train, cv_folds=5):
    """Cross-validation for XGBoost models"""
    print("\n" + "="*60)
    print("XGBOOST CROSS-VALIDATION")
    print("="*60)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_results = {}

    for name, model in models_dict.items():
        print(f"\nCross-validating {name}...")

        # Multiple scoring metrics
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

        scores = {}
        for metric in scoring_metrics:
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv, scoring=metric, n_jobs=-1
            )
            scores[metric] = {
                'mean': cv_scores.mean(),
                'std': cv_scores.std(),
                'scores': cv_scores
            }
            print(f"  {metric}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        cv_results[name] = scores

    return cv_results

def create_xgboost_visualizations(models_results, feature_importance_results, output_dir):
    """Create XGBoost-specific visualizations"""
    print("\n" + "="*60)
    print("CREATING XGBOOST VISUALIZATIONS")
    print("="*60)

    plots_dir = Path(output_dir) / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8')

    # 1. Model comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        model_names = list(models_results.keys())
        train_vals = [models_results[name]['train'][metric] for name in model_names]
        val_vals = [models_results[name]['validation'][metric] for name in model_names]
        test_vals = [models_results[name]['test'][metric] for name in model_names]

        x = np.arange(len(model_names))
        width = 0.25

        ax.bar(x - width, train_vals, width, label='Train', alpha=0.8)
        ax.bar(x, val_vals, width, label='Validation', alpha=0.8)
        ax.bar(x + width, test_vals, width, label='Test', alpha=0.8)

        ax.set_xlabel('Models')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'XGBoost {metric.capitalize()} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Remove empty subplot
    fig.delaxes(axes[-1])
    plt.tight_layout()
    plt.savefig(plots_dir / 'xgboost_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Feature importance visualization
    if feature_importance_results:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        for idx, (importance_type, importance_df) in enumerate(feature_importance_results.items()):
            ax = axes[idx]

            # Plot top 15 features
            top_features = importance_df.head(15)
            bars = ax.barh(top_features['feature'], top_features[f'{importance_type}_importance'])

            ax.set_xlabel(f'{importance_type.capitalize()} Importance')
            ax.set_title(f'Top 15 Features by {importance_type.capitalize()}')
            ax.invert_yaxis()

            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2,
                       f'{width:.3f}', ha='left', va='center')

        plt.tight_layout()
        plt.savefig(plots_dir / 'xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

    print(f"✓ XGBoost visualizations saved in {plots_dir}/")

def train_xgboost_models(datasets):
    """Train multiple XGBoost variants"""
    print("\n" + "="*60)
    print("TRAINING XGBOOST MODELS")
    print("="*60)

    X_train = datasets['X_train']
    X_val = datasets['X_val']
    X_test = datasets['X_test']
    y_train = datasets['y_train']
    y_val = datasets['y_val']
    y_test = datasets['y_test']

    X_train_smote = datasets['X_train_smote']
    y_train_smote = datasets['y_train_smote']
    class_weights = datasets['class_weights']

    # Calculate sample weights from class weights
    sample_weights = np.array([class_weights[0] if y == 0 else class_weights[1] for y in y_train])

    models = {}
    results = {}

    # 1. Basic XGBoost
    print("\n1. Training Basic XGBoost...")
    xgb_basic = xgb.XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric='auc'
    )

    eval_set = [(X_val, y_val)]
    xgb_basic.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    models['Basic_XGB'] = xgb_basic

    # 2. XGBoost with scale_pos_weight (for imbalanced data)
    print("\n2. Training XGBoost with scale_pos_weight...")
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_balanced = xgb.XGBClassifier(
        n_estimators=100,
        scale_pos_weight=pos_weight,
        random_state=42,
        eval_metric='auc'
    )

    xgb_balanced.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    models['Balanced_XGB'] = xgb_balanced

    # 3. XGBoost with SMOTE data
    print("\n3. Training XGBoost with SMOTE data...")
    xgb_smote = xgb.XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric='auc'
    )

    eval_set_smote = [(X_val, y_val)]
    xgb_smote.fit(X_train_smote, y_train_smote, eval_set=eval_set_smote, verbose=False)
    models['SMOTE_XGB'] = xgb_smote

    # 4. Hyperparameter-tuned XGBoost models
    print("\n4. Hyperparameter Optimization...")

    # Tune on balanced data
    print("Tuning XGBoost with scale_pos_weight...")
    best_balanced_model, best_balanced_params, study_balanced = perform_optuna_optimization(
        X_train, y_train, X_val, y_val, sample_weight=None, n_trials=50
    )
    # Update with scale_pos_weight
    best_balanced_params['scale_pos_weight'] = pos_weight
    best_balanced_model_final = xgb.XGBClassifier(**best_balanced_params)
    best_balanced_model_final.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    models['Best_Balanced_XGB'] = best_balanced_model_final

    # Tune on SMOTE data
    print("Tuning XGBoost with SMOTE data...")
    best_smote_model, best_smote_params, study_smote = perform_optuna_optimization(
        X_train_smote, y_train_smote, X_val, y_val, sample_weight=None, n_trials=50
    )
    models['Best_SMOTE_XGB'] = best_smote_model

    # Evaluate all models
    print("\n" + "="*60)
    print("XGBOOST MODEL EVALUATION")
    print("="*60)

    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}:")
        results[model_name] = {}

        # Use appropriate training data for evaluation
        if 'SMOTE' in model_name:
            train_X, train_y = X_train_smote, y_train_smote
        else:
            train_X, train_y = X_train, y_train

        results[model_name]['train'] = evaluate_xgboost_performance(
            model, train_X, train_y, "Training"
        )
        results[model_name]['validation'] = evaluate_xgboost_performance(
            model, X_val, y_val, "Validation"
        )
        results[model_name]['test'] = evaluate_xgboost_performance(
            model, X_test, y_test, "Test"
        )

    # Cross-validation
    cv_results = cross_validate_xgboost(models, X_train, y_train)

    # Feature importance for best model
    best_model_name = max(models.keys(),
                         key=lambda x: results[x]['validation']['auc'])
    best_model = models[best_model_name]

    print(f"\n🏆 Best XGBoost model: {best_model_name}")
    print(f"Best validation AUC: {results[best_model_name]['validation']['auc']:.4f}")

    feature_names = X_train.columns.tolist()
    feature_importance_results = analyze_xgboost_feature_importance(best_model, feature_names)

    return models, results, cv_results, feature_importance_results, best_model_name

def save_xgboost_results(models, results, cv_results, feature_importance, best_model_name, output_dir):
    """Save all XGBoost results"""
    print("\n" + "="*60)
    print("SAVING XGBOOST RESULTS")
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

    cv_file = output_path / 'cv_results.pkl'
    with open(cv_file, 'wb') as f:
        pickle.dump(cv_results, f)

    # Save feature importance
    for importance_type, importance_df in feature_importance.items():
        importance_file = output_path / f'feature_importance_{importance_type}.csv'
        importance_df.to_csv(importance_file, index=False)

    # Save best model
    best_model_info = {
        'best_model_name': best_model_name,
        'best_model': models[best_model_name],
        'performance': results[best_model_name]
    }

    best_model_file = output_path / 'best_xgboost_model.pkl'
    with open(best_model_file, 'wb') as f:
        pickle.dump(best_model_info, f)

    print(f"✓ All XGBoost results saved to: {output_path}")

def main():
    """Main XGBoost modeling pipeline"""
    print("STARTING XGBOOST MODEL DEVELOPMENT")
    print("="*80)

    # Configuration
    DATA_DIR = '02_preprocessing/processed_data'
    OUTPUT_DIR = '04_xgboost_model/results'

    # Load data
    datasets = load_processed_data(DATA_DIR)

    # Train and evaluate models
    models, results, cv_results, feature_importance, best_model_name = train_xgboost_models(datasets)

    # Create visualizations
    create_xgboost_visualizations(results, feature_importance, OUTPUT_DIR)

    # Save results
    save_xgboost_results(models, results, cv_results, feature_importance, best_model_name, OUTPUT_DIR)

    # Summary
    print("\n" + "="*60)
    print("XGBOOST MODELING SUMMARY")
    print("="*60)
    print(f"✓ Models trained: {len(models)}")
    print(f"✓ Best model: {best_model_name}")
    print(f"✓ Best validation AUC: {results[best_model_name]['validation']['auc']:.4f}")
    print(f"✓ Results saved to: {OUTPUT_DIR}")

    return models, results, best_model_name

if __name__ == "__main__":
    models, results, best_model_name = main()