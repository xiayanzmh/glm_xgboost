#!/usr/bin/env python3
"""
Data Exploration Script
GLM vs XGBoost Modeling Project

This script performs comprehensive exploratory data analysis on the modeling dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

def load_and_inspect_data(filepath):
    """Load data and perform basic inspection"""
    print("="*60)
    print("LOADING AND INSPECTING DATA")
    print("="*60)

    # Load data
    df = pd.read_excel(filepath)

    # Basic info
    print(f"Data loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")

    return df

def analyze_missing_values(df):
    """Analyze missing values in the dataset"""
    print("\n" + "="*60)
    print("MISSING VALUES ANALYSIS")
    print("="*60)

    missing_data = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) * 100
    missing_table = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent
    })
    missing_table = missing_table[missing_table['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

    if len(missing_table) > 0:
        print(missing_table)
    else:
        print("✓ No missing values found!")

    return missing_table

def analyze_target_variable(df, target_col='Y'):
    """Analyze the target variable"""
    print("\n" + "="*60)
    print("TARGET VARIABLE ANALYSIS")
    print("="*60)

    print(f"Target variable '{target_col}' statistics:")
    print(df[target_col].describe())

    unique_values = df[target_col].nunique()
    print(f"\nUnique values: {unique_values}")

    # Determine problem type
    if unique_values <= 20:
        problem_type = 'classification'
        print(f"*** CLASSIFICATION problem detected (distinct values: {unique_values}) ***")
        print(f"Value counts:\n{df[target_col].value_counts().sort_index()}")
    else:
        problem_type = 'regression'
        print(f"*** REGRESSION problem detected (distinct values: {unique_values}) ***")
        print(f"Value range: {df[target_col].min():.4f} to {df[target_col].max():.4f}")

    # Statistical measures
    print(f"\nSkewness: {df[target_col].skew():.4f}")
    print(f"Kurtosis: {df[target_col].kurtosis():.4f}")

    return problem_type

def analyze_features(df):
    """Analyze feature variables"""
    print("\n" + "="*60)
    print("FEATURE ANALYSIS")
    print("="*60)

    feature_cols = [col for col in df.columns if col.startswith('V')]
    print(f"Number of features: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols}")

    # Feature statistics
    feature_stats = df[feature_cols].describe()
    print("\nFeature statistics:")
    print(feature_stats)

    return feature_cols, feature_stats

def create_visualizations(df, target_col='Y', output_dir='01_eda'):
    """Create and save visualizations"""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)

    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    feature_cols = [col for col in df.columns if col.startswith('V')]

    # 1. Target variable distribution
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(df[target_col], bins=50, alpha=0.7, edgecolor='black')
    plt.title(f'Distribution of {target_col}')
    plt.xlabel(target_col)
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    plt.boxplot(df[target_col])
    plt.title(f'Box Plot of {target_col}')
    plt.ylabel(target_col)

    plt.subplot(1, 3, 3)
    from scipy.stats import probplot
    probplot(df[target_col], dist="norm", plot=plt)
    plt.title(f'Q-Q Plot of {target_col}')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'target_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Feature distributions
    n_features = len(feature_cols)
    n_cols = 5
    n_rows = (n_features + n_cols - 1) // n_cols

    plt.figure(figsize=(20, 4 * n_rows))
    for i, feature in enumerate(feature_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.hist(df[feature], bins=30, alpha=0.7, edgecolor='black')
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Correlation matrix
    plt.figure(figsize=(20, 16))
    corr_matrix = df[feature_cols + [target_col]].corr()

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 4. Feature vs Target relationships
    plt.figure(figsize=(20, 4 * n_rows))
    for i, feature in enumerate(feature_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.scatter(df[feature], df[target_col], alpha=0.5)
        plt.xlabel(feature)
        plt.ylabel(target_col)
        plt.title(f'{feature} vs {target_col}')

        # Add correlation coefficient
        corr = df[feature].corr(df[target_col])
        plt.text(0.05, 0.95, f'r = {corr:.3f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_target_relationships.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✓ Visualizations saved in {plots_dir}/")

def identify_outliers(df, target_col='Y'):
    """Identify outliers in the dataset"""
    print("\n" + "="*60)
    print("OUTLIER ANALYSIS")
    print("="*60)

    feature_cols = [col for col in df.columns if col.startswith('V')]
    # Only include numeric columns for outlier analysis
    numeric_cols = [col for col in feature_cols + [target_col] if df[col].dtype in ['int64', 'float64']]
    categorical_cols = [col for col in feature_cols if df[col].dtype == 'object']

    if categorical_cols:
        print(f"Categorical features found: {categorical_cols}")
        print("Analyzing categorical feature distribution:")
        for col in categorical_cols:
            print(f"\n{col} value counts:")
            print(df[col].value_counts().head(10))

    outlier_summary = {}

    print(f"\nAnalyzing outliers for numeric features...")
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(df)) * 100

        outlier_summary[col] = {
            'count': outlier_count,
            'percentage': outlier_percentage,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }

    # Create outlier summary table
    outlier_df = pd.DataFrame(outlier_summary).T
    outlier_df = outlier_df[outlier_df['count'] > 0].sort_values('count', ascending=False)

    if len(outlier_df) > 0:
        print("Outliers detected (using IQR method):")
        print(outlier_df)
    else:
        print("✓ No outliers detected using IQR method")

    return outlier_summary

def generate_data_quality_report(df, target_col='Y'):
    """Generate comprehensive data quality report"""
    print("\n" + "="*60)
    print("DATA QUALITY REPORT")
    print("="*60)

    report = {
        'total_records': len(df),
        'total_features': len([col for col in df.columns if col.startswith('V')]),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_records': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }

    print(f"Total records: {report['total_records']:,}")
    print(f"Total features: {report['total_features']}")
    print(f"Missing values: {report['missing_values']}")
    print(f"Duplicate records: {report['duplicate_records']}")
    print(f"Memory usage: {report['memory_usage_mb']:.2f} MB")

    # Data types summary
    print(f"\nData type distribution:")
    print(df.dtypes.value_counts())

    return report

def main():
    """Main execution function"""
    # File path
    data_path = '../data/Data.xlsx'

    # Load and inspect data
    df = load_and_inspect_data(data_path)

    # Analyze missing values
    missing_analysis = analyze_missing_values(df)

    # Analyze target variable
    problem_type = analyze_target_variable(df)

    # Analyze features
    feature_cols, feature_stats = analyze_features(df)

    # Identify outliers
    outlier_analysis = identify_outliers(df)

    # Generate data quality report
    quality_report = generate_data_quality_report(df)

    # Create visualizations
    create_visualizations(df)

    # Save summary report
    summary = {
        'problem_type': problem_type,
        'data_shape': df.shape,
        'feature_count': len(feature_cols),
        'missing_values': len(missing_analysis) > 0,
        'quality_report': quality_report
    }

    # Save processed data info for next steps
    print("\n" + "="*60)
    print("EXPLORATION COMPLETE")
    print("="*60)
    print("Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    return df, summary

if __name__ == "__main__":
    df, summary = main()