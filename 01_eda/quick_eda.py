#!/usr/bin/env python3
"""
Quick EDA Script for GLM vs XGBoost Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def main():
    # Load data
    print("Loading data...")
    df = pd.read_excel('data/Data.xlsx')

    print(f"Data shape: {df.shape}")
    print(f"Data types:\n{df.dtypes}")

    # Basic info
    print("\n" + "="*50)
    print("BASIC DATA INFORMATION")
    print("="*50)
    print(f"Total records: {len(df):,}")
    print(f"Features: V1-V18 (18 features)")
    print(f"Target: Y (binary classification)")
    print(f"Missing values: {df.isnull().sum().sum()}")

    # Target analysis
    print("\n" + "="*50)
    print("TARGET VARIABLE ANALYSIS")
    print("="*50)
    target_counts = df['Y'].value_counts().sort_index()
    print(f"Class distribution:\n{target_counts}")
    print(f"Class proportions:\n{df['Y'].value_counts(normalize=True).sort_index()}")

    # Feature types
    print("\n" + "="*50)
    print("FEATURE ANALYSIS")
    print("="*50)
    numeric_features = []
    categorical_features = []

    for col in df.columns:
        if col.startswith('V'):
            if df[col].dtype in ['object', 'str', str]:
                categorical_features.append(col)
                print(f"{col} (categorical): {df[col].nunique()} unique values")
                print(f"  Top values: {df[col].value_counts().head(3).to_dict()}")
            else:
                numeric_features.append(col)

    print(f"\nNumeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    # Numeric features summary
    if numeric_features:
        print(f"\nNumeric features summary:")
        numeric_summary = df[numeric_features].describe()
        print(numeric_summary)

    # Correlation with target
    print("\n" + "="*50)
    print("FEATURE-TARGET CORRELATIONS")
    print("="*50)
    if numeric_features:
        # Only use numeric features for correlation
        numeric_df = df[numeric_features + ['Y']]
        correlations = numeric_df.corr()['Y'].sort_values(key=abs, ascending=False)[1:]  # Exclude Y-Y correlation
        print("Top correlations with target:")
        print(correlations.head(10))

    # Check for potential issues
    print("\n" + "="*50)
    print("DATA QUALITY CHECKS")
    print("="*50)

    # Duplicate rows
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")

    # Zero variance features
    zero_var_features = []
    for col in numeric_features:
        if df[col].std() == 0:
            zero_var_features.append(col)

    if zero_var_features:
        print(f"Zero variance features: {zero_var_features}")
    else:
        print("✓ No zero variance features")

    # High cardinality categorical features
    high_card_features = []
    for col in categorical_features:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.1:  # More than 10% unique values
            high_card_features.append((col, df[col].nunique(), unique_ratio))

    if high_card_features:
        print("High cardinality categorical features:")
        for col, nunique, ratio in high_card_features:
            print(f"  {col}: {nunique} unique ({ratio:.3f} ratio)")

    # Summary for next steps
    print("\n" + "="*50)
    print("SUMMARY FOR MODELING")
    print("="*50)
    print(f"✓ Dataset: {len(df):,} records, {len(df.columns)} columns")
    print(f"✓ Problem: Binary classification (imbalanced: {target_counts[0]:,} vs {target_counts[1]:,})")
    print(f"✓ Features: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
    print(f"✓ Data quality: {df.isnull().sum().sum()} missing values, {duplicates} duplicates")

    # Modeling recommendations
    print("\nRecommendations:")
    print("1. Handle class imbalance (consider SMOTE, class weights, or stratified sampling)")
    print("2. Encode categorical variables (V10)")
    print("3. Scale numerical features for GLM")
    print("4. Consider feature engineering and selection")

    # Create output directory for pipeline completion tracking
    output_base = os.environ.get('OUTPUT_BASE_DIR', '.')
    plots_dir = Path(output_base) / '01_eda' / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save EDA summary for pipeline tracking
    summary_file = plots_dir / 'eda_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("=== Quick EDA Summary ===\n")
        f.write(f"Dataset shape: {df.shape}\n")
        f.write(f"Missing values: {df.isnull().sum().sum()}\n")
        f.write(f"Target distribution:\n{df['Y'].value_counts().to_string()}\n")
        f.write(f"Numeric features: {len([col for col in df.columns if col.startswith('V') and df[col].dtype in ['int64', 'float64']])}\n")
        f.write(f"Categorical features: {len([col for col in df.columns if col.startswith('V') and df[col].dtype == 'object'])}\n")

    print(f"\n📁 EDA summary saved to: {summary_file}")

    return df

if __name__ == "__main__":
    df = main()