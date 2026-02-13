#!/usr/bin/env python3
"""
Data Preprocessing Pipeline
GLM vs XGBoost Modeling Project

This script handles:
1. Data loading and basic preprocessing
2. Categorical variable encoding
3. Feature scaling for GLM
4. Train/validation/test splits with stratification
5. Class imbalance handling options
6. Feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import pickle
import os
from pathlib import Path

def load_data(filepath):
    df = pd.read_excel(filepath)
    print(f"Data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Target distribution:\n{df['Y'].value_counts().sort_index()}")

    return df

def encode_categorical_features(df):
    print("CATEGORICAL FEATURE ENCODING")

    df_processed = df.copy()

    # Encode V10 (A/B) to (0/1)
    if 'V10' in df_processed.columns:
        print(f"V10 original distribution:\n{df_processed['V10'].value_counts()}")

        # Create label encoder
        le_v10 = LabelEncoder()
        df_processed['V10'] = le_v10.fit_transform(df_processed['V10'])

        print(f"V10 encoded distribution:\n{df_processed['V10'].value_counts().sort_index()}")
        print(f"Encoding mapping: {dict(zip(le_v10.classes_, range(len(le_v10.classes_))))}")

    return df_processed, le_v10

def feature_engineering(df):
    """Create additional features"""
    print("\n" + "="*60)
    print("FEATURE ENGINEERING")
    print("="*60)

    df_engineered = df.copy()

    # Get original feature columns
    feature_cols = [col for col in df.columns if col.startswith('V')]

    # 1. Interaction features (top correlated features)
    # Based on EDA: V13, V3, V7 are top correlated
    df_engineered['V13_V3_interaction'] = df['V13'] * df['V3']
    df_engineered['V13_V7_interaction'] = df['V13'] * df['V7']
    df_engineered['V3_V7_interaction'] = df['V3'] * df['V7']

    # 2. Polynomial features for top predictors
    df_engineered['V13_squared'] = df['V13'] ** 2
    df_engineered['V3_squared'] = df['V3'] ** 2

    # 3. Ratios (for interpretability)
    # Avoid division by zero
    df_engineered['V2_V18_ratio'] = df['V2'] / (df['V18'] + 1)
    df_engineered['V13_V5_ratio'] = df['V13'] / (df['V5'] + 1)

    # 4. Binned features (for non-linear relationships)
    df_engineered['V13_binned'] = pd.cut(df['V13'], bins=5, labels=[0,1,2,3,4])
    df_engineered['V3_binned'] = pd.cut(df['V3'], bins=5, labels=[0,1,2,3,4])

    # Convert binned features to numeric
    df_engineered['V13_binned'] = df_engineered['V13_binned'].astype(int)
    df_engineered['V3_binned'] = df_engineered['V3_binned'].astype(int)

    # 5. Aggregate features
    numeric_features = [col for col in feature_cols if col != 'V10']
    df_engineered['feature_sum'] = df[numeric_features].sum(axis=1)
    df_engineered['feature_mean'] = df[numeric_features].mean(axis=1)
    df_engineered['feature_std'] = df[numeric_features].std(axis=1)

    new_features = [col for col in df_engineered.columns if col not in df.columns]
    print(f"Created {len(new_features)} new features: {new_features}")

    return df_engineered

def create_data_splits(df, test_size=0.2, val_size=0.15, random_state=42):
    """Create train/validation/test splits with stratification"""
    print("\n" + "="*60)
    print("CREATING DATA SPLITS")
    print("="*60)

    # Separate features and target
    X = df.drop(['ID', 'Y'], axis=1)
    y = df['Y']

    print(f"Features: {X.shape[1]} columns")
    print(f"Target distribution: {y.value_counts().sort_index().to_dict()}")

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Second split: train vs val
    # Adjust validation size relative to remaining data
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=random_state
    )

    print(f"\nData splits created:")
    print(f"Train: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
    print(f"Validation: {X_val.shape[0]:,} samples ({X_val.shape[0]/len(df)*100:.1f}%)")
    print(f"Test: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(df)*100:.1f}%)")

    # Check stratification
    print(f"\nTarget distribution in splits:")
    print(f"Train: {y_train.value_counts().sort_index().to_dict()}")
    print(f"Val: {y_val.value_counts().sort_index().to_dict()}")
    print(f"Test: {y_test.value_counts().sort_index().to_dict()}")

    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_features_for_glm(X_train, X_val, X_test):
    """Scale features for GLM (logistic regression)"""
    print("\n" + "="*60)
    print("FEATURE SCALING FOR GLM")
    print("="*60)

    # Initialize scaler
    scaler = StandardScaler()

    # Fit on training data only
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    print(f"Features scaled: {X_train_scaled.shape[1]} columns")
    print(f"Train scaled shape: {X_train_scaled.shape}")
    print(f"Validation scaled shape: {X_val_scaled.shape}")
    print(f"Test scaled shape: {X_test_scaled.shape}")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def handle_class_imbalance(X_train, y_train, method='smote', random_state=42):
    """Handle class imbalance using various techniques"""
    print("\n" + "="*60)
    print(f"HANDLING CLASS IMBALANCE: {method.upper()}")
    print("="*60)

    print(f"Original class distribution: {y_train.value_counts().sort_index().to_dict()}")

    if method == 'smote':
        sampler = SMOTE(random_state=random_state)
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

    elif method == 'smote_tomek':
        sampler = SMOTETomek(random_state=random_state)
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

    elif method == 'undersample':
        sampler = RandomUnderSampler(random_state=random_state)
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

    elif method == 'class_weights':
        # Calculate class weights for use in models
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        print(f"Calculated class weights: {class_weight_dict}")
        return X_train, y_train, class_weight_dict

    elif method == 'none':
        print("No resampling applied")
        return X_train, y_train, None

    else:
        raise ValueError(f"Unknown method: {method}")

    if method != 'class_weights':
        # Convert back to DataFrame/Series
        X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
        y_resampled = pd.Series(y_resampled, name=y_train.name)

        print(f"Resampled class distribution: {y_resampled.value_counts().sort_index().to_dict()}")
        print(f"Resampled size: {len(X_resampled):,} samples")

        return X_resampled, y_resampled, None

def save_processed_data(data_dict, output_dir):
    """Save all processed data and preprocessors"""
    print("\n" + "="*60)
    print("SAVING PROCESSED DATA")
    print("="*60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save datasets
    for name, data in data_dict.items():
        if isinstance(data, (pd.DataFrame, pd.Series)):
            filepath = output_path / f"{name}.pkl"
            data.to_pickle(filepath)
            print(f"✓ Saved {name}: {filepath}")
        elif data is not None:
            filepath = output_path / f"{name}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            print(f"✓ Saved {name}: {filepath}")

    print(f"\nAll processed data saved to: {output_path}")

def main():
    """Main preprocessing pipeline"""
    print("STARTING DATA PREPROCESSING PIPELINE")
    print("="*80)

    # Configuration
    DATA_PATH = 'data/Data.xlsx'

    # Use timestamped output directory if provided by pipeline
    output_base = os.environ.get('OUTPUT_BASE_DIR', '.')
    OUTPUT_DIR = Path(output_base) / '02_preprocessing' / 'processed_data'

    RANDOM_STATE = 42

    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Encode categorical features
    df_encoded, label_encoder = encode_categorical_features(df)

    # 3. Feature engineering
    df_engineered = feature_engineering(df_encoded)

    # 4. Create data splits
    X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(
        df_engineered, random_state=RANDOM_STATE
    )

    # 5. Scale features for GLM
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features_for_glm(
        X_train, X_val, X_test
    )

    # 6. Handle class imbalance (create multiple versions)
    # SMOTE version
    X_train_smote, y_train_smote, _ = handle_class_imbalance(
        X_train, y_train, method='smote', random_state=RANDOM_STATE
    )
    X_train_scaled_smote, _, _, _ = scale_features_for_glm(
        X_train_smote, X_val, X_test  # Only refit scaler on SMOTE data
    )

    # Class weights version
    _, _, class_weights = handle_class_imbalance(
        X_train, y_train, method='class_weights'
    )

    # 7. Save all processed data
    processed_data = {
        # Original splits
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,

        # Scaled versions (for GLM)
        'X_train_scaled': X_train_scaled,
        'X_val_scaled': X_val_scaled,
        'X_test_scaled': X_test_scaled,

        # SMOTE versions
        'X_train_smote': X_train_smote,
        'y_train_smote': y_train_smote,
        'X_train_scaled_smote': X_train_scaled_smote,

        # Preprocessors
        'scaler': scaler,
        'label_encoder': label_encoder,
        'class_weights': class_weights,

        # Original data for reference
        'df_original': df,
        'df_engineered': df_engineered
    }

    save_processed_data(processed_data, OUTPUT_DIR)

    # Summary
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"✓ Original data: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"✓ Engineered data: {df_engineered.shape[0]:,} rows × {df_engineered.shape[1]} columns")
    print(f"✓ Training set: {X_train.shape[0]:,} rows × {X_train.shape[1]} columns")
    print(f"✓ Class imbalance handled: SMOTE and class weights")
    print(f"✓ Features scaled for GLM")
    print(f"✓ All data saved to: {OUTPUT_DIR}")

    return processed_data

if __name__ == "__main__":
    processed_data = main()