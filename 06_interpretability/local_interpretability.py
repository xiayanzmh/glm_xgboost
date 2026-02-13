#!/usr/bin/env python3
"""
Local Interpretability Analysis
GLM vs XGBoost Modeling Project

This script provides local interpretability analysis for the winning model using:
1. LIME (Local Interpretable Model-agnostic Explanations)
2. Feature attribution for individual predictions
3. Visualization of local explanations
4. Analysis of different instance types (correct/incorrect predictions)
5. Feature importance stability analysis
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path

from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

def load_winner_model_and_data():
    """Load the winning model and test data for interpretability analysis"""
    print("="*60)
    print("LOADING WINNER MODEL AND DATA")
    print("="*60)

    # Load comparison results to determine winner
    comparison_results_path = Path('05_model_comparison/results/model_comparison_results.pkl')

    if comparison_results_path.exists():
        with open(comparison_results_path, 'rb') as f:
            comparison_results = pickle.load(f)
        winner_name = comparison_results['winner']
        print(f"✓ Winner model: {winner_name}")
    else:
        # Fallback - load both models and compare manually
        print("⚠ Comparison results not found. Loading GLM as default for demo...")
        winner_name = 'GLM'  # Default for demonstration

    # Load appropriate model and data
    if winner_name.upper() == 'GLM':
        # Load GLM model and scaled data
        glm_path = Path('03_glm_model/results/best_glm_model.pkl')
        with open(glm_path, 'rb') as f:
            model_info = pickle.load(f)

        model = model_info['best_model']
        model_type = 'GLM'

        # Load scaled test data for GLM
        data_path = Path('02_preprocessing/processed_data')
        with open(data_path / 'X_test_scaled.pkl', 'rb') as f:
            X_test = pickle.load(f)
        with open(data_path / 'X_train_scaled.pkl', 'rb') as f:
            X_train = pickle.load(f)

    else:  # XGBoost winner
        # Load XGBoost model and unscaled data
        xgb_path = Path('04_xgboost_model/results/best_xgboost_model.pkl')
        with open(xgb_path, 'rb') as f:
            model_info = pickle.load(f)

        model = model_info['best_model']
        model_type = 'XGBoost'

        # Load unscaled test data for XGBoost
        data_path = Path('02_preprocessing/processed_data')
        with open(data_path / 'X_test.pkl', 'rb') as f:
            X_test = pickle.load(f)
        with open(data_path / 'X_train.pkl', 'rb') as f:
            X_train = pickle.load(f)

    # Load target
    with open(data_path / 'y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)

    print(f"✓ Model type: {model_type}")
    print(f"✓ Test data shape: {X_test.shape}")
    print(f"✓ Feature names: {list(X_test.columns)}")

    return model, model_type, X_train, X_test, y_test

def setup_lime_explainer(model, X_train, model_type):
    """Setup LIME explainer for the model"""
    print("\n" + "="*60)
    print("SETTING UP LIME EXPLAINER")
    print("="*60)

    # Convert to numpy arrays if they're DataFrames
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f'Feature_{i}' for i in range(X_train_np.shape[1])]

    # Create LIME explainer
    explainer = LimeTabularExplainer(
        X_train_np,
        feature_names=feature_names,
        class_names=['Class_0', 'Class_1'],
        mode='classification',
        discretize_continuous=True,
        random_state=42
    )

    print(f"✓ LIME explainer created for {model_type}")
    print(f"✓ Training data shape: {X_train_np.shape}")
    print(f"✓ Number of features: {len(feature_names)}")

    return explainer, feature_names

def explain_individual_predictions(model, explainer, X_test, y_test, feature_names, n_samples=10):
    """Generate explanations for individual predictions"""
    print("\n" + "="*60)
    print("GENERATING INDIVIDUAL EXPLANATIONS")
    print("="*60)

    # Convert to numpy if DataFrame
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test

    # Get model predictions
    predictions = model.predict(X_test_np)
    probabilities = model.predict_proba(X_test_np)

    # Select diverse examples for explanation
    indices_to_explain = []

    # Get some correct predictions for each class
    correct_class_0 = np.where((predictions == 0) & (y_test == 0))[0]
    correct_class_1 = np.where((predictions == 1) & (y_test == 1))[0]

    # Get some incorrect predictions
    incorrect_class_0 = np.where((predictions == 1) & (y_test == 0))[0]  # False Positives
    incorrect_class_1 = np.where((predictions == 0) & (y_test == 1))[0]  # False Negatives

    # Sample from each category
    for category, indices in [
        ('Correct Class 0', correct_class_0),
        ('Correct Class 1', correct_class_1),
        ('False Positive', incorrect_class_0),
        ('False Negative', incorrect_class_1)
    ]:
        if len(indices) > 0:
            sample_size = min(3, len(indices))  # Take up to 3 examples from each category
            sampled_indices = np.random.choice(indices, sample_size, replace=False)
            indices_to_explain.extend(sampled_indices)

    # Limit total number of explanations
    indices_to_explain = indices_to_explain[:n_samples]

    explanations = []
    explanation_data = []

    print(f"Generating explanations for {len(indices_to_explain)} instances...")

    for i, idx in enumerate(indices_to_explain):
        instance = X_test_np[idx]
        true_label = y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]
        predicted_label = predictions[idx]
        prediction_proba = probabilities[idx]

        # Generate explanation
        explanation = explainer.explain_instance(
            instance,
            model.predict_proba,
            num_features=len(feature_names),
            num_samples=1000
        )

        explanations.append(explanation)

        # Extract explanation data
        exp_data = {
            'instance_id': idx,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'prediction_probability': prediction_proba[1],  # Probability of class 1
            'explanation': explanation,
            'feature_importance': dict(explanation.as_list())
        }

        explanation_data.append(exp_data)

        print(f"  Instance {i+1}: True={true_label}, Pred={predicted_label}, Prob={prediction_proba[1]:.3f}")

    return explanations, explanation_data

def analyze_feature_stability(explanation_data, feature_names):
    """Analyze stability of feature importance across explanations"""
    print("\n" + "="*60)
    print("ANALYZING FEATURE IMPORTANCE STABILITY")
    print("="*60)

    # Extract feature importance from all explanations
    feature_importance_matrix = []

    for exp_data in explanation_data:
        feature_importance = exp_data['feature_importance']

        # Create importance vector for all features
        importance_vector = []
        for feature in feature_names:
            # LIME might return feature values with conditions, extract feature name
            importance = 0
            for lime_feature, importance_val in feature_importance.items():
                if feature in lime_feature:  # Simple matching
                    importance = importance_val
                    break
            importance_vector.append(importance)

        feature_importance_matrix.append(importance_vector)

    # Convert to DataFrame
    importance_df = pd.DataFrame(feature_importance_matrix, columns=feature_names)

    # Calculate stability metrics
    stability_metrics = {
        'mean_importance': importance_df.mean(),
        'std_importance': importance_df.std(),
        'coefficient_of_variation': importance_df.std() / (importance_df.mean().abs() + 1e-8)
    }

    print("Feature importance stability analysis:")
    stability_summary = pd.DataFrame({
        'Mean_Importance': stability_metrics['mean_importance'],
        'Std_Importance': stability_metrics['std_importance'],
        'Coeff_Variation': stability_metrics['coefficient_of_variation']
    })

    # Sort by absolute mean importance
    stability_summary['Abs_Mean_Importance'] = stability_summary['Mean_Importance'].abs()
    stability_summary = stability_summary.sort_values('Abs_Mean_Importance', ascending=False)

    print(stability_summary.head(15))

    return importance_df, stability_metrics

def create_interpretability_visualizations(explanations, explanation_data, importance_df, output_dir):
    """Create comprehensive interpretability visualizations"""
    print("\n" + "="*60)
    print("CREATING INTERPRETABILITY VISUALIZATIONS")
    print("="*60)

    plots_dir = Path(output_dir) / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8')

    # 1. Individual explanation examples
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    # Show first 4 explanations
    for i, exp_data in enumerate(explanation_data[:4]):
        if i >= 4:
            break

        ax = axes[i]
        explanation = exp_data['explanation']
        feature_importance = exp_data['feature_importance']

        # Get top features (positive and negative)
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

        features = [item[0] for item in sorted_features]
        importances = [item[1] for item in sorted_features]

        # Create color map (positive = green, negative = red)
        colors = ['green' if imp > 0 else 'red' for imp in importances]

        bars = ax.barh(range(len(features)), importances, color=colors, alpha=0.7)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels([f.split(' ')[0][:15] + '...' if len(f) > 15 else f.split(' ')[0] for f in features])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Instance {exp_data["instance_id"]} (True: {exp_data["true_label"]}, '
                    f'Pred: {exp_data["predicted_label"]}, Prob: {exp_data["prediction_probability"]:.3f})')
        ax.grid(True, alpha=0.3)

        # Add value labels
        for bar, imp in zip(bars, importances):
            ax.text(bar.get_width() + (0.01 if imp > 0 else -0.01), bar.get_y() + bar.get_height()/2,
                   f'{imp:.3f}', ha='left' if imp > 0 else 'right', va='center')

    plt.tight_layout()
    plt.savefig(plots_dir / 'individual_explanations.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Feature importance stability heatmap
    plt.figure(figsize=(16, 12))

    # Select top 20 features by mean absolute importance for visualization
    mean_abs_importance = importance_df.abs().mean()
    top_features = mean_abs_importance.nlargest(20).index

    # Create heatmap of feature importance across instances
    importance_subset = importance_df[top_features]

    sns.heatmap(importance_subset.T, cmap='RdBu_r', center=0, annot=False,
                cbar_kws={'label': 'Feature Importance'})
    plt.title('Feature Importance Across Explained Instances\n(Top 20 Features)')
    plt.xlabel('Instance ID')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(plots_dir / 'feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Feature importance distribution
    plt.figure(figsize=(16, 10))

    # Select top 15 features for distribution plots
    top_features = mean_abs_importance.nlargest(15).index

    n_features = len(top_features)
    n_cols = 5
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for i, feature in enumerate(top_features):
        ax = axes[i]
        feature_values = importance_df[feature]

        ax.hist(feature_values, bins=15, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax.set_title(f'{feature[:20]}...' if len(feature) > 20 else feature)
        ax.set_xlabel('Importance')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

    # Remove empty subplots
    for i in range(len(top_features), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(plots_dir / 'feature_importance_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 4. Prediction confidence vs explanation clarity
    plt.figure(figsize=(12, 8))

    confidences = [exp_data['prediction_probability'] for exp_data in explanation_data]
    explanation_clarity = []

    # Calculate explanation clarity as the sum of absolute feature importances
    for exp_data in explanation_data:
        clarity = sum(abs(imp) for imp in exp_data['feature_importance'].values())
        explanation_clarity.append(clarity)

    plt.scatter(confidences, explanation_clarity, alpha=0.7)
    plt.xlabel('Prediction Confidence (Probability of Positive Class)')
    plt.ylabel('Explanation Clarity (Sum of Absolute Feature Importance)')
    plt.title('Prediction Confidence vs Explanation Clarity')
    plt.grid(True, alpha=0.3)

    # Add correlation coefficient
    corr = np.corrcoef(confidences, explanation_clarity)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.savefig(plots_dir / 'confidence_vs_clarity.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✓ Interpretability visualizations saved in {plots_dir}/")

def generate_interpretability_report(model_type, explanation_data, stability_metrics, output_dir):
    """Generate comprehensive interpretability report"""
    print("\n" + "="*60)
    print("GENERATING INTERPRETABILITY REPORT")
    print("="*60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_file = output_path / 'interpretability_report.txt'

    with open(report_file, 'w') as f:
        f.write("LOCAL INTERPRETABILITY ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")

        f.write(f"MODEL TYPE: {model_type}\n")
        f.write(f"ANALYSIS DATE: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"NUMBER OF EXPLAINED INSTANCES: {len(explanation_data)}\n\n")

        f.write("SUMMARY STATISTICS:\n")
        f.write("-" * 20 + "\n")

        # Calculate summary statistics
        confidences = [exp['prediction_probability'] for exp in explanation_data]
        correct_predictions = [exp['true_label'] == exp['predicted_label'] for exp in explanation_data]

        f.write(f"Average prediction confidence: {np.mean(confidences):.3f}\n")
        f.write(f"Prediction accuracy in sample: {np.mean(correct_predictions):.3f}\n")
        f.write(f"Confidence range: [{min(confidences):.3f}, {max(confidences):.3f}]\n\n")

        f.write("TOP 10 MOST STABLE FEATURES:\n")
        f.write("-" * 30 + "\n")

        # Get most stable features (low coefficient of variation, high importance)
        stability_df = pd.DataFrame({
            'Feature': stability_metrics['mean_importance'].index,
            'Mean_Importance': stability_metrics['mean_importance'].values,
            'Coeff_Variation': stability_metrics['coefficient_of_variation'].values
        })

        # Filter for important features and sort by stability
        important_features = stability_df[stability_df['Mean_Importance'].abs() > 0.01]
        stable_features = important_features.sort_values('Coeff_Variation').head(10)

        for idx, row in stable_features.iterrows():
            f.write(f"{row['Feature']}: Importance = {row['Mean_Importance']:.3f}, "
                   f"Stability = {1/row['Coeff_Variation']:.1f}\n")

        f.write("\n\nINDIVIDUAL INSTANCE ANALYSIS:\n")
        f.write("-" * 30 + "\n")

        for i, exp_data in enumerate(explanation_data):
            f.write(f"\nInstance {i+1} (ID: {exp_data['instance_id']}):\n")
            f.write(f"  True Label: {exp_data['true_label']}\n")
            f.write(f"  Predicted Label: {exp_data['predicted_label']}\n")
            f.write(f"  Prediction Confidence: {exp_data['prediction_probability']:.3f}\n")

            # Top 3 positive and negative features
            sorted_features = sorted(exp_data['feature_importance'].items(),
                                   key=lambda x: x[1], reverse=True)

            f.write(f"  Top Supporting Features:\n")
            for feature, importance in sorted_features[:3]:
                if importance > 0:
                    f.write(f"    {feature.split(' ')[0]}: +{importance:.3f}\n")

            f.write(f"  Top Opposing Features:\n")
            for feature, importance in sorted_features[-3:]:
                if importance < 0:
                    f.write(f"    {feature.split(' ')[0]}: {importance:.3f}\n")

    print(f"✓ Interpretability report saved to: {report_file}")

def save_interpretability_results(explanations, explanation_data, importance_df, stability_metrics, output_dir):
    """Save all interpretability results"""
    print("\n" + "="*60)
    print("SAVING INTERPRETABILITY RESULTS")
    print("="*60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save explanations
    explanations_file = output_path / 'lime_explanations.pkl'
    with open(explanations_file, 'wb') as f:
        pickle.dump(explanations, f)

    # Save explanation data
    explanation_data_file = output_path / 'explanation_data.pkl'
    with open(explanation_data_file, 'wb') as f:
        pickle.dump(explanation_data, f)

    # Save feature importance matrix
    importance_file = output_path / 'feature_importance_matrix.csv'
    importance_df.to_csv(importance_file, index=False)

    # Save stability metrics
    stability_file = output_path / 'stability_metrics.pkl'
    with open(stability_file, 'wb') as f:
        pickle.dump(stability_metrics, f)

    print(f"✓ All interpretability results saved to: {output_path}")

def main():
    """Main interpretability analysis pipeline"""
    print("STARTING LOCAL INTERPRETABILITY ANALYSIS")
    print("="*80)

    # Configuration
    OUTPUT_DIR = '06_interpretability/results'

    # Load winner model and data
    model, model_type, X_train, X_test, y_test = load_winner_model_and_data()

    # Setup LIME explainer
    explainer, feature_names = setup_lime_explainer(model, X_train, model_type)

    # Generate individual explanations
    explanations, explanation_data = explain_individual_predictions(
        model, explainer, X_test, y_test, feature_names, n_samples=12
    )

    # Analyze feature stability
    importance_df, stability_metrics = analyze_feature_stability(explanation_data, feature_names)

    # Create visualizations
    create_interpretability_visualizations(explanations, explanation_data, importance_df, OUTPUT_DIR)

    # Generate report
    generate_interpretability_report(model_type, explanation_data, stability_metrics, OUTPUT_DIR)

    # Save results
    save_interpretability_results(explanations, explanation_data, importance_df, stability_metrics, OUTPUT_DIR)

    print("\n" + "="*60)
    print("INTERPRETABILITY ANALYSIS SUMMARY")
    print("="*60)
    print(f"✓ Model analyzed: {model_type}")
    print(f"✓ Instances explained: {len(explanation_data)}")
    print(f"✓ Features analyzed: {len(feature_names)}")
    print(f"✓ Results saved to: {OUTPUT_DIR}")

    return explanation_data, importance_df

if __name__ == "__main__":
    explanation_data, importance_df = main()