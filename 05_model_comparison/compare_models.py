#!/usr/bin/env python3
"""
Model Comparison: GLM vs XGBoost
GLM vs XGBoost Modeling Project

This script compares the performance of GLM and XGBoost models:
1. Load best models from both approaches
2. Comprehensive performance comparison
3. Statistical significance testing
4. Visualization of results
5. Model selection based on multiple criteria
6. Business impact analysis
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

def load_model_results(glm_dir, xgboost_dir):
    """Load results from both GLM and XGBoost models"""
    print("="*60)
    print("LOADING MODEL RESULTS")
    print("="*60)

    results = {}

    # Load GLM results
    glm_path = Path(glm_dir)
    with open(glm_path / 'best_glm_model.pkl', 'rb') as f:
        glm_results = pickle.load(f)

    with open(glm_path / 'evaluation_results.pkl', 'rb') as f:
        glm_eval_results = pickle.load(f)

    results['GLM'] = {
        'best_model': glm_results['best_model'],
        'best_model_name': glm_results['best_model_name'],
        'performance': glm_results['performance'],
        'all_results': glm_eval_results
    }

    print(f"✓ GLM Best Model: {glm_results['best_model_name']}")
    print(f"  Validation AUC: {glm_results['performance']['validation']['auc']:.4f}")

    # Load XGBoost results
    xgboost_path = Path(xgboost_dir)
    with open(xgboost_path / 'best_xgboost_model.pkl', 'rb') as f:
        xgb_results = pickle.load(f)

    with open(xgboost_path / 'evaluation_results.pkl', 'rb') as f:
        xgb_eval_results = pickle.load(f)

    results['XGBoost'] = {
        'best_model': xgb_results['best_model'],
        'best_model_name': xgb_results['best_model_name'],
        'performance': xgb_results['performance'],
        'all_results': xgb_eval_results
    }

    print(f"✓ XGBoost Best Model: {xgb_results['best_model_name']}")
    print(f"  Validation AUC: {xgb_results['performance']['validation']['auc']:.4f}")

    return results

def load_test_data():
    """Load test dataset for final evaluation"""
    data_path = Path('02_preprocessing/processed_data')

    # Load test data (both scaled and unscaled)
    with open(data_path / 'X_test_scaled.pkl', 'rb') as f:
        X_test_scaled = pickle.load(f)

    with open(data_path / 'X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)

    with open(data_path / 'y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)

    return X_test_scaled, X_test, y_test

def perform_final_evaluation(models, X_test_scaled, X_test, y_test):
    """Perform final evaluation on test set"""
    print("\n" + "="*60)
    print("FINAL MODEL EVALUATION ON TEST SET")
    print("="*60)

    final_results = {}

    # GLM evaluation (uses scaled data)
    glm_model = models['GLM']['best_model']
    glm_pred = glm_model.predict(X_test_scaled)
    glm_pred_proba = glm_model.predict_proba(X_test_scaled)[:, 1]

    # XGBoost evaluation (uses unscaled data)
    xgb_model = models['XGBoost']['best_model']
    xgb_pred = xgb_model.predict(X_test)
    xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

    # Calculate metrics for both models
    for model_name, y_pred, y_pred_proba in [('GLM', glm_pred, glm_pred_proba),
                                             ('XGBoost', xgb_pred, xgb_pred_proba)]:

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }

        final_results[model_name] = {
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

        print(f"\n{model_name} Test Performance:")
        for metric, value in metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")

    return final_results

def statistical_significance_test(results1, results2, y_test):
    """Test statistical significance between model performances"""
    print("\n" + "="*60)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("="*60)

    # AUC comparison using bootstrapping
    n_bootstrap = 1000
    auc_diff_bootstrap = []

    for i in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(len(y_test), len(y_test), replace=True)
        y_boot = y_test.iloc[indices]
        prob1_boot = results1['probabilities'][indices]
        prob2_boot = results2['probabilities'][indices]

        auc1 = roc_auc_score(y_boot, prob1_boot)
        auc2 = roc_auc_score(y_boot, prob2_boot)
        auc_diff_bootstrap.append(auc2 - auc1)

    auc_diff_mean = np.mean(auc_diff_bootstrap)
    auc_diff_std = np.std(auc_diff_bootstrap)

    # Calculate confidence interval
    ci_lower = np.percentile(auc_diff_bootstrap, 2.5)
    ci_upper = np.percentile(auc_diff_bootstrap, 97.5)

    print(f"AUC Difference (XGBoost - GLM): {auc_diff_mean:.4f} ± {auc_diff_std:.4f}")
    print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")

    if ci_lower > 0:
        print("✓ XGBoost significantly better than GLM (p < 0.05)")
        significant_winner = "XGBoost"
    elif ci_upper < 0:
        print("✓ GLM significantly better than XGBoost (p < 0.05)")
        significant_winner = "GLM"
    else:
        print("⚠ No significant difference between models (p > 0.05)")
        significant_winner = "No significant difference"

    return {
        'auc_difference_mean': auc_diff_mean,
        'auc_difference_std': auc_diff_std,
        'confidence_interval': (ci_lower, ci_upper),
        'significant_winner': significant_winner,
        'bootstrap_differences': auc_diff_bootstrap
    }

def create_comparison_visualizations(final_results, stat_test_results, y_test, output_dir):
    """Create comprehensive comparison visualizations"""
    print("\n" + "="*60)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("="*60)

    plots_dir = Path(output_dir) / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8')

    # 1. Performance metrics comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    glm_values = [final_results['GLM']['metrics'][m] for m in metrics]
    xgb_values = [final_results['XGBoost']['metrics'][m] for m in metrics]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        models = ['GLM', 'XGBoost']
        values = [glm_values[idx], xgb_values[idx]]
        colors = ['skyblue', 'lightcoral']

        bars = ax.bar(models, values, color=colors, alpha=0.8)
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_ylim(0, 1)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        ax.grid(True, alpha=0.3)

    # Remove empty subplot
    fig.delaxes(axes[-1])
    plt.tight_layout()
    plt.savefig(plots_dir / 'model_comparison_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. ROC Curves comparison
    plt.figure(figsize=(10, 8))

    # GLM ROC
    fpr_glm, tpr_glm, _ = roc_curve(y_test, final_results['GLM']['probabilities'])
    auc_glm = final_results['GLM']['metrics']['auc']

    # XGBoost ROC
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, final_results['XGBoost']['probabilities'])
    auc_xgb = final_results['XGBoost']['metrics']['auc']

    plt.plot(fpr_glm, tpr_glm, linewidth=2, label=f'GLM (AUC = {auc_glm:.3f})', color='blue')
    plt.plot(fpr_xgb, tpr_xgb, linewidth=2, label=f'XGBoost (AUC = {auc_xgb:.3f})', color='red')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(plots_dir / 'roc_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Precision-Recall curves comparison
    plt.figure(figsize=(10, 8))

    # GLM PR curve
    precision_glm, recall_glm, _ = precision_recall_curve(y_test, final_results['GLM']['probabilities'])

    # XGBoost PR curve
    precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, final_results['XGBoost']['probabilities'])

    plt.plot(recall_glm, precision_glm, linewidth=2, label=f'GLM', color='blue')
    plt.plot(recall_xgb, precision_xgb, linewidth=2, label=f'XGBoost', color='red')

    # Baseline (random classifier)
    baseline = (y_test == 1).mean()
    plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1, label=f'Random Classifier (P={baseline:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(plots_dir / 'pr_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 4. Confusion matrices comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (model_name, results) in enumerate([('GLM', final_results['GLM']),
                                                ('XGBoost', final_results['XGBoost'])]):
        cm = results['confusion_matrix']
        ax = axes[idx]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{model_name} Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig(plots_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 5. AUC difference bootstrap distribution
    plt.figure(figsize=(10, 6))
    plt.hist(stat_test_results['bootstrap_differences'], bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Difference')
    plt.axvline(x=stat_test_results['auc_difference_mean'], color='blue', linestyle='-', linewidth=2,
                label=f'Observed Difference ({stat_test_results["auc_difference_mean"]:.4f})')

    ci_lower, ci_upper = stat_test_results['confidence_interval']
    plt.axvline(x=ci_lower, color='green', linestyle=':', linewidth=2, alpha=0.7)
    plt.axvline(x=ci_upper, color='green', linestyle=':', linewidth=2, alpha=0.7,
                label=f'95% CI [{ci_lower:.4f}, {ci_upper:.4f}]')

    plt.xlabel('AUC Difference (XGBoost - GLM)')
    plt.ylabel('Frequency')
    plt.title('Bootstrap Distribution of AUC Differences')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(plots_dir / 'auc_difference_bootstrap.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✓ Comparison visualizations saved in {plots_dir}/")

def model_selection_analysis(final_results, stat_test_results):
    """Comprehensive model selection analysis"""
    print("\n" + "="*60)
    print("MODEL SELECTION ANALYSIS")
    print("="*60)

    # Create scoring matrix
    criteria = {
        'Test AUC': {'GLM': final_results['GLM']['metrics']['auc'],
                    'XGBoost': final_results['XGBoost']['metrics']['auc'],
                    'weight': 0.3, 'higher_better': True},

        'Test F1-Score': {'GLM': final_results['GLM']['metrics']['f1'],
                         'XGBoost': final_results['XGBoost']['metrics']['f1'],
                         'weight': 0.25, 'higher_better': True},

        'Test Precision': {'GLM': final_results['GLM']['metrics']['precision'],
                          'XGBoost': final_results['XGBoost']['metrics']['precision'],
                          'weight': 0.2, 'higher_better': True},

        'Test Recall': {'GLM': final_results['GLM']['metrics']['recall'],
                       'XGBoost': final_results['XGBoost']['metrics']['recall'],
                       'weight': 0.15, 'higher_better': True},

        'Interpretability': {'GLM': 0.9, 'XGBoost': 0.6,  # GLM more interpretable
                           'weight': 0.1, 'higher_better': True}
    }

    # Calculate weighted scores
    scores = {'GLM': 0, 'XGBoost': 0}

    print(f"{'Criterion':<20} {'GLM':<10} {'XGBoost':<10} {'Weight':<10} {'Winner':<15}")
    print("-" * 70)

    for criterion, data in criteria.items():
        glm_val = data['GLM']
        xgb_val = data['XGBoost']
        weight = data['weight']

        # Normalize scores (0-1 scale)
        max_val = max(glm_val, xgb_val)
        min_val = min(glm_val, xgb_val)

        if max_val != min_val:
            glm_norm = (glm_val - min_val) / (max_val - min_val)
            xgb_norm = (xgb_val - min_val) / (max_val - min_val)
        else:
            glm_norm = xgb_norm = 1.0

        # Add to weighted scores
        scores['GLM'] += glm_norm * weight
        scores['XGBoost'] += xgb_norm * weight

        # Determine winner for this criterion
        winner = 'GLM' if glm_val > xgb_val else 'XGBoost' if xgb_val > glm_val else 'Tie'

        print(f"{criterion:<20} {glm_val:<10.4f} {xgb_val:<10.4f} {weight:<10.2f} {winner:<15}")

    print("-" * 70)
    print(f"{'TOTAL SCORE':<20} {scores['GLM']:<10.4f} {scores['XGBoost']:<10.4f}")

    # Final recommendation
    if scores['XGBoost'] > scores['GLM']:
        recommended_model = 'XGBoost'
        score_difference = scores['XGBoost'] - scores['GLM']
    else:
        recommended_model = 'GLM'
        score_difference = scores['GLM'] - scores['XGBoost']

    print(f"\n🏆 RECOMMENDED MODEL: {recommended_model}")
    print(f"Score difference: {score_difference:.4f}")

    # Additional considerations
    print(f"\nADDITIONAL CONSIDERATIONS:")
    print(f"• Statistical significance: {stat_test_results['significant_winner']}")

    if stat_test_results['significant_winner'] != 'No significant difference':
        print(f"• AUC difference: {stat_test_results['auc_difference_mean']:.4f} ± {stat_test_results['auc_difference_std']:.4f}")

    return {
        'recommended_model': recommended_model,
        'scores': scores,
        'score_difference': score_difference,
        'criteria_analysis': criteria
    }

def save_comparison_results(final_results, stat_test_results, selection_analysis, output_dir):
    """Save all comparison results"""
    print("\n" + "="*60)
    print("SAVING COMPARISON RESULTS")
    print("="*60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save comprehensive results
    comparison_results = {
        'final_test_results': final_results,
        'statistical_significance': stat_test_results,
        'model_selection_analysis': selection_analysis,
        'winner': selection_analysis['recommended_model']
    }

    results_file = output_path / 'model_comparison_results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(comparison_results, f)

    # Save summary report
    summary_file = output_path / 'model_comparison_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("GLM vs XGBoost Model Comparison Summary\n")
        f.write("="*50 + "\n\n")

        f.write("TEST SET PERFORMANCE:\n")
        f.write("-"*20 + "\n")
        for model, results in final_results.items():
            f.write(f"\n{model}:\n")
            for metric, value in results['metrics'].items():
                f.write(f"  {metric.capitalize()}: {value:.4f}\n")

        f.write(f"\n\nSTATISTICAL SIGNIFICANCE:\n")
        f.write("-"*25 + "\n")
        f.write(f"Winner: {stat_test_results['significant_winner']}\n")
        f.write(f"AUC Difference: {stat_test_results['auc_difference_mean']:.4f}\n")
        f.write(f"95% CI: {stat_test_results['confidence_interval']}\n")

        f.write(f"\n\nRECOMMENDED MODEL: {selection_analysis['recommended_model']}\n")
        f.write(f"Score Difference: {selection_analysis['score_difference']:.4f}\n")

    print(f"✓ Comparison results saved to: {output_path}")

def main():
    """Main model comparison pipeline"""
    print("STARTING MODEL COMPARISON")
    print("="*80)

    # Configuration
    GLM_DIR = '03_glm_model/results'
    XGBOOST_DIR = '04_xgboost_model/results'
    OUTPUT_DIR = '05_model_comparison/results'

    # Load model results
    model_results = load_model_results(GLM_DIR, XGBOOST_DIR)

    # Load test data
    X_test_scaled, X_test, y_test = load_test_data()

    # Final evaluation
    final_results = perform_final_evaluation(model_results, X_test_scaled, X_test, y_test)

    # Statistical significance testing
    stat_test_results = statistical_significance_test(
        final_results['GLM'], final_results['XGBoost'], y_test
    )

    # Model selection analysis
    selection_analysis = model_selection_analysis(final_results, stat_test_results)

    # Create visualizations
    create_comparison_visualizations(final_results, stat_test_results, y_test, OUTPUT_DIR)

    # Save results
    save_comparison_results(final_results, stat_test_results, selection_analysis, OUTPUT_DIR)

    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"🏆 Winner: {selection_analysis['recommended_model']}")
    print(f"📊 GLM Test AUC: {final_results['GLM']['metrics']['auc']:.4f}")
    print(f"📊 XGBoost Test AUC: {final_results['XGBoost']['metrics']['auc']:.4f}")
    print(f"📈 Statistical significance: {stat_test_results['significant_winner']}")

    return selection_analysis['recommended_model'], final_results

if __name__ == "__main__":
    winner, final_results = main()