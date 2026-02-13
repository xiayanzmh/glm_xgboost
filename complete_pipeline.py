#!/usr/bin/env python3
"""
Complete ML Pipeline - GLM vs XGBoost
All-in-one machine learning pipeline for binary classification

Usage:
    python complete_pipeline.py --all                    # Run entire pipeline
    python complete_pipeline.py --phase eda             # Run only EDA
    python complete_pipeline.py --phase preprocessing   # Run only preprocessing
    python complete_pipeline.py --phase modeling        # Run both GLM and XGBoost
    python complete_pipeline.py --phase comparison      # Run model comparison
"""

import argparse
import os
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ML imports
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# XGBoost
import xgboost as xgb

# Imbalance handling
from imblearn.over_sampling import SMOTE

# Interpretability
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

warnings.filterwarnings('ignore')

class CompletePipeline:
    def __init__(self):
        """Initialize the complete ML pipeline"""
        # Set up timestamped output directory
        self.setup_output_directory()

        # Data containers
        self.df_original = None
        self.df_engineered = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = None
        self.label_encoder = None

        # Models
        self.glm_model = None
        self.xgb_model = None

        # Results
        self.results = {}


    def setup_output_directory(self):
        """Set up timestamped output directory"""
        today = datetime.now().strftime('%Y-%m-%d')
        self.output_dir = Path(f'outputs/{today}')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        for subdir in ['eda', 'preprocessing', 'models', 'results', 'plots']:
            (self.output_dir / subdir).mkdir(exist_ok=True)

        # Create symlink to latest run
        latest_link = Path('outputs/latest')
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        try:
            latest_link.symlink_to(today, target_is_directory=True)
        except (OSError, NotImplementedError):
            pass

    def phase_1_eda(self):
        print("PHASE 1: EXPLORATORY DATA ANALYSIS")


        # Load data
        print("Loading data...")
        self.df_original = pd.read_excel('data/Data.xlsx')

        print(f"Data shape: {self.df_original.shape}")
        print(f"Data types:\n{self.df_original.dtypes}")

        # Basic information
        print(f"\nBasic Information:")
        print(f"- Total records: {len(self.df_original):,}")
        print(f"- Features: {len([col for col in self.df_original.columns if col.startswith('V')])} features")
        print(f"- Target: Y (binary classification)")
        print(f"- Missing values: {self.df_original.isnull().sum().sum()}")

        # Target analysis
        print(f"\nTarget Variable Analysis:")
        target_counts = self.df_original['Y'].value_counts().sort_index()
        print(f"Class distribution:\n{target_counts}")
        print(f"Class proportions:\n{self.df_original['Y'].value_counts(normalize=True).sort_index()}")

        # Feature analysis
        numeric_features = []
        categorical_features = []

        for col in self.df_original.columns:
            if col.startswith('V'):
                if self.df_original[col].dtype in ['object', 'str', str]:
                    categorical_features.append(col)
                    print(f"{col} (categorical): {self.df_original[col].nunique()} unique values")
                else:
                    numeric_features.append(col)

        print(f"\nFeature Summary:")
        print(f"- Numeric features: {len(numeric_features)}")
        print(f"- Categorical features: {len(categorical_features)}")

        # Correlations
        if numeric_features:
            correlations = self.df_original[numeric_features + ['Y']].corr()['Y'].sort_values(key=abs, ascending=False)[1:]
            print(f"\nTop correlations with target:")
            print(correlations.head(10))

        # Save EDA summary
        eda_summary_path = self.output_dir / 'eda' / 'eda_summary.txt'
        with open(eda_summary_path, 'w') as f:
            f.write("=== EDA Summary ===\n")
            f.write(f"Dataset shape: {self.df_original.shape}\n")
            f.write(f"Missing values: {self.df_original.isnull().sum().sum()}\n")
            f.write(f"Target distribution:\n{target_counts.to_string()}\n")
            f.write(f"Numeric features: {len(numeric_features)}\n")
            f.write(f"Categorical features: {len(categorical_features)}\n")

        print(f"\nEDA summary saved to: {eda_summary_path}")
        print("Phase 1 (EDA) completed successfully!")

    def phase_2_preprocessing(self):
        """Phase 2: Data Preprocessing & Feature Engineering"""
        print("\n" + "="*80)
        print("PHASE 2: DATA PREPROCESSING & FEATURE ENGINEERING")
        print("="*80)

        # Load data if not already loaded
        if self.df_original is None:
            print("Loading original data...")
            try:
                self.df_original = pd.read_excel('data/Data.xlsx')
                print(f"Data loaded: {self.df_original.shape}")
            except Exception as e:
                print(f"Error loading data: {e}")
                return False

        # Copy data for processing
        df = self.df_original.copy()

        # Categorical encoding
        print("\nCategorical Feature Encoding:")
        categorical_features = [col for col in df.columns if col.startswith('V') and df[col].dtype == 'object']

        self.label_encoder = LabelEncoder()
        for col in categorical_features:
            print(f"Encoding {col}...")
            df[col] = self.label_encoder.fit_transform(df[col])

        # Feature Engineering
        print("\nFeature Engineering:")
        feature_cols = [col for col in df.columns if col.startswith('V')]

        # Create interaction features
        if 'V13' in feature_cols and 'V3' in feature_cols:
            df['V13_V3_interaction'] = df['V13'] * df['V3']
            print("Created V13_V3_interaction")

        if 'V13' in feature_cols and 'V7' in feature_cols:
            df['V13_V7_interaction'] = df['V13'] * df['V7']
            print("Created V13_V7_interaction")

        # Create polynomial features for top correlated features
        if 'V13' in feature_cols:
            df['V13_squared'] = df['V13'] ** 2
            print("Created V13_squared")

        if 'V3' in feature_cols:
            df['V3_squared'] = df['V3'] ** 2
            print("Created V3_squared")

        # Create ratio features
        if 'V2' in feature_cols and 'V18' in feature_cols:
            df['V2_V18_ratio'] = df['V2'] / (df['V18'] + 1)  # Add 1 to avoid division by zero
            print("Created V2_V18_ratio")

        # Create aggregate features
        df['feature_sum'] = df[feature_cols].sum(axis=1)
        df['feature_mean'] = df[feature_cols].mean(axis=1)
        df['feature_std'] = df[feature_cols].std(axis=1)
        print("Created aggregate features: sum, mean, std")

        self.df_engineered = df

        # Prepare features and target
        X = df.drop(['ID', 'Y'], axis=1, errors='ignore')
        y = df['Y']

        print(f"\nFeatures: {X.shape[1]} columns")
        print(f"Target distribution: {y.value_counts().sort_index().to_dict()}")

        # Create data splits
        print("\nCreating Data Splits:")
        # First split: train+val vs test (80/20)
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # Second split: train vs val (65/15 of total, which is 81.25/18.75 of temp)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.1875, random_state=42, stratify=y_temp
        )

        print(f"Train: {self.X_train.shape[0]:,} samples ({self.X_train.shape[0]/len(df)*100:.1f}%)")
        print(f"Validation: {self.X_val.shape[0]:,} samples ({self.X_val.shape[0]/len(df)*100:.1f}%)")
        print(f"Test: {self.X_test.shape[0]:,} samples ({self.X_test.shape[0]/len(df)*100:.1f}%)")

        # Feature Scaling
        print("\nFeature Scaling:")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_val_scaled = self.scaler.transform(self.X_val)
        X_test_scaled = self.scaler.transform(self.X_test)

        # Convert back to DataFrames
        self.X_train = pd.DataFrame(X_train_scaled, columns=self.X_train.columns)
        self.X_val = pd.DataFrame(X_val_scaled, columns=self.X_val.columns)
        self.X_test = pd.DataFrame(X_test_scaled, columns=self.X_test.columns)

        print(f"Features scaled: {self.X_train.shape[1]} columns")

        # Handle class imbalance with SMOTE
        print("\nHandling Class Imbalance with SMOTE:")
        print(f"Original class distribution: {self.y_train.value_counts().sort_index().to_dict()}")

        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(self.X_train, self.y_train)

        # Store SMOTE data for models that can benefit from it
        self.X_train_smote = pd.DataFrame(X_train_smote, columns=self.X_train.columns)
        self.y_train_smote = pd.Series(y_train_smote, name=self.y_train.name)

        print(f"Resampled class distribution: {self.y_train_smote.value_counts().sort_index().to_dict()}")
        print(f"Resampled size: {len(self.X_train_smote):,} samples")

        # Calculate class weights for models that support them
        self.class_weights = compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)
        self.class_weight_dict = {0: self.class_weights[0], 1: self.class_weights[1]}
        print(f"Calculated class weights: {self.class_weight_dict}")

        # Save processed data
        preprocessing_dir = self.output_dir / 'preprocessing'

        # Save datasets
        self.X_train.to_pickle(preprocessing_dir / 'X_train.pkl')
        self.X_val.to_pickle(preprocessing_dir / 'X_val.pkl')
        self.X_test.to_pickle(preprocessing_dir / 'X_test.pkl')
        self.y_train.to_pickle(preprocessing_dir / 'y_train.pkl')
        self.y_val.to_pickle(preprocessing_dir / 'y_val.pkl')
        self.y_test.to_pickle(preprocessing_dir / 'y_test.pkl')
        self.X_train_smote.to_pickle(preprocessing_dir / 'X_train_smote.pkl')
        self.y_train_smote.to_pickle(preprocessing_dir / 'y_train_smote.pkl')

        # Save preprocessors
        with open(preprocessing_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(preprocessing_dir / 'label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        with open(preprocessing_dir / 'class_weights.pkl', 'wb') as f:
            pickle.dump(self.class_weight_dict, f)

        print(f"\nAll processed data saved to: {preprocessing_dir}")
        print("Phase 2 (Preprocessing) completed successfully!")
        return True

    def phase_3_glm_modeling(self):
        """Phase 3: GLM Model Development"""
        print("\n" + "="*80)
        print("PHASE 3: GLM MODEL DEVELOPMENT")
        print("="*80)

        if self.X_train is None:
            print("Error: No processed data available. Run preprocessing phase first.")
            return False

        # GLM with hyperparameter tuning
        print("\nGLM Hyperparameter Tuning:")

        # Use SMOTE data for GLM training
        X_train_use = self.X_train_smote
        y_train_use = self.y_train_smote

        # Parameter grid for GLM
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'saga'],
            'class_weight': [None, 'balanced']
        }

        # Create GLM model
        glm = LogisticRegression(random_state=42, max_iter=1000)

        # Grid search
        print("Running GridSearchCV for GLM...")
        grid_search = GridSearchCV(
            glm, param_grid, cv=5, scoring='roc_auc',
            n_jobs=-1, verbose=1
        )

        grid_search.fit(X_train_use, y_train_use)

        self.glm_model = grid_search.best_estimator_

        print(f"Best GLM parameters: {grid_search.best_params_}")
        print(f"Best GLM CV score: {grid_search.best_score_:.4f}")

        # Evaluate GLM
        print("\nGLM Model Evaluation:")
        self.results['glm'] = self.evaluate_model(self.glm_model, 'GLM')

        # Save GLM model
        models_dir = self.output_dir / 'models'
        with open(models_dir / 'glm_model.pkl', 'wb') as f:
            pickle.dump(self.glm_model, f)

        print("Phase 3 (GLM) completed successfully!")
        return True

    def phase_4_xgboost_modeling(self):
        """Phase 4: XGBoost Model Development"""
        print("\n" + "="*80)
        print("PHASE 4: XGBOOST MODEL DEVELOPMENT")
        print("="*80)

        if self.X_train is None:
            print("Error: No processed data available. Run preprocessing phase first.")
            return False

        # XGBoost with hyperparameter tuning
        print("\nXGBoost Hyperparameter Tuning:")

        # Use original training data with class weights for XGBoost
        X_train_use = self.X_train
        y_train_use = self.y_train

        # Calculate scale_pos_weight for XGBoost
        scale_pos_weight = len(y_train_use[y_train_use == 0]) / len(y_train_use[y_train_use == 1])

        # Parameter grid for XGBoost
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'scale_pos_weight': [1, scale_pos_weight]
        }

        # Create XGBoost model
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )

        # Grid search with smaller param grid due to computational cost
        print("Running GridSearchCV for XGBoost...")
        # Use a smaller grid for faster execution
        simplified_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.1, 0.2],
            'scale_pos_weight': [1, scale_pos_weight]
        }

        grid_search = GridSearchCV(
            xgb_model, simplified_grid, cv=3, scoring='roc_auc',
            n_jobs=-1, verbose=1
        )

        grid_search.fit(X_train_use, y_train_use)

        self.xgb_model = grid_search.best_estimator_

        print(f"Best XGBoost parameters: {grid_search.best_params_}")
        print(f"Best XGBoost CV score: {grid_search.best_score_:.4f}")

        # Evaluate XGBoost
        print("\nXGBoost Model Evaluation:")
        self.results['xgboost'] = self.evaluate_model(self.xgb_model, 'XGBoost')

        # Save XGBoost model
        models_dir = self.output_dir / 'models'
        with open(models_dir / 'xgboost_model.pkl', 'wb') as f:
            pickle.dump(self.xgb_model, f)

        print("Phase 4 (XGBoost) completed successfully!")
        return True

    def evaluate_model(self, model, model_name):
        """Evaluate a model on train, validation, and test sets"""
        results = {'model_name': model_name}

        for split_name, X_data, y_data in [
            ('train', self.X_train, self.y_train),
            ('validation', self.X_val, self.y_val),
            ('test', self.X_test, self.y_test)
        ]:
            # Make predictions
            y_pred = model.predict(X_data)
            y_pred_proba = model.predict_proba(X_data)[:, 1]

            # Calculate metrics
            results[f'{split_name}_accuracy'] = accuracy_score(y_data, y_pred)
            results[f'{split_name}_precision'] = precision_score(y_data, y_pred)
            results[f'{split_name}_recall'] = recall_score(y_data, y_pred)
            results[f'{split_name}_f1'] = f1_score(y_data, y_pred)
            results[f'{split_name}_auc'] = roc_auc_score(y_data, y_pred_proba)

            print(f"{split_name.capitalize()} Results:")
            print(f"  Accuracy: {results[f'{split_name}_accuracy']:.4f}")
            print(f"  Precision: {results[f'{split_name}_precision']:.4f}")
            print(f"  Recall: {results[f'{split_name}_recall']:.4f}")
            print(f"  F1-Score: {results[f'{split_name}_f1']:.4f}")
            print(f"  AUC: {results[f'{split_name}_auc']:.4f}")

        return results

    def phase_5_model_comparison(self):
        """Phase 5: Model Comparison & Selection"""
        print("\n" + "="*80)
        print("PHASE 5: MODEL COMPARISON & SELECTION")
        print("="*80)

        if not self.results:
            print("Error: No model results available. Run modeling phases first.")
            return False

        print("\nModel Performance Comparison:")
        print("="*50)

        # Create comparison table
        comparison_data = []
        metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_auc']

        for model_name, results in self.results.items():
            row = {'Model': results['model_name']}
            for metric in metrics:
                if metric in results:
                    row[metric.replace('test_', '').upper()] = f"{results[metric]:.4f}"
            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))

        # Determine best model based on AUC
        best_model_name = None
        best_auc = 0

        for model_name, results in self.results.items():
            if results.get('test_auc', 0) > best_auc:
                best_auc = results['test_auc']
                best_model_name = model_name

        print(f"\nBest Model: {best_model_name.upper()} (AUC: {best_auc:.4f})")

        # Save comparison results
        results_dir = self.output_dir / 'results'
        comparison_df.to_csv(results_dir / 'model_comparison.csv', index=False)

        # Save detailed results
        with open(results_dir / 'detailed_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)

        print("Phase 5 (Model Comparison) completed successfully!")
        return True

    def phase_6_interpretability(self):
        """Phase 6: Model Interpretability Analysis"""
        print("\n" + "="*80)
        print("PHASE 6: MODEL INTERPRETABILITY ANALYSIS")
        print("="*80)

        if not LIME_AVAILABLE:
            print("LIME not available. Skipping interpretability analysis.")
            print("Install with: pip install lime")
            return False

        if self.glm_model is None or self.xgb_model is None:
            print("Error: Models not available. Run modeling phases first.")
            return False

        print("\nGenerating LIME explanations...")

        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_train.values,
            feature_names=self.X_train.columns.tolist(),
            class_names=['Class 0', 'Class 1'],
            mode='classification'
        )

        # Generate explanations for a sample of test instances
        sample_indices = np.random.choice(len(self.X_test), size=min(10, len(self.X_test)), replace=False)

        explanations = {}

        for model_name, model in [('GLM', self.glm_model), ('XGBoost', self.xgb_model)]:
            explanations[model_name] = []

            for i, idx in enumerate(sample_indices):
                instance = self.X_test.iloc[idx].values

                # Generate explanation
                exp = explainer.explain_instance(
                    instance,
                    model.predict_proba,
                    num_features=10
                )

                explanations[model_name].append({
                    'instance_id': idx,
                    'prediction': model.predict_proba([instance])[0],
                    'explanation': exp.as_list()
                })

        # Save explanations
        results_dir = self.output_dir / 'results'
        with open(results_dir / 'lime_explanations.pkl', 'wb') as f:
            pickle.dump(explanations, f)

        print(f"LIME explanations generated for {len(sample_indices)} test instances")
        print("Phase 6 (Interpretability) completed successfully!")
        return True

    def phase_7_documentation(self):
        """Phase 7: Final Documentation"""
        print("\n" + "="*80)
        print("PHASE 7: FINAL DOCUMENTATION")
        print("="*80)

        # Generate final report
        report_path = self.output_dir / 'final_report.txt'

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GLM vs XGBoost Pipeline - Final Report\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Dataset summary
            f.write("DATASET SUMMARY:\n")
            f.write(f"- Original shape: {self.df_original.shape if self.df_original is not None else 'N/A'}\n")
            f.write(f"- Engineered shape: {self.df_engineered.shape if self.df_engineered is not None else 'N/A'}\n")
            f.write(f"- Training samples: {len(self.X_train) if self.X_train is not None else 'N/A'}\n")
            f.write(f"- Test samples: {len(self.X_test) if self.X_test is not None else 'N/A'}\n\n")

            # Model results
            f.write("MODEL PERFORMANCE:\n")
            if self.results:
                for model_name, results in self.results.items():
                    f.write(f"\n{results['model_name']}:\n")
                    f.write(f"  Test AUC: {results.get('test_auc', 'N/A'):.4f}\n")
                    f.write(f"  Test Accuracy: {results.get('test_accuracy', 'N/A'):.4f}\n")
                    f.write(f"  Test F1-Score: {results.get('test_f1', 'N/A'):.4f}\n")
            else:
                f.write("No model results available.\n")

            f.write(f"\nOutput directory: {self.output_dir}\n")
            f.write("Pipeline completed successfully!\n")

        print(f"Final report saved to: {report_path}")
        print("Phase 7 (Documentation) completed successfully!")
        return True

    def run_phase(self, phase_name):
        """Run a specific phase"""
        phase_methods = {
            'eda': self.phase_1_eda,
            'preprocessing': self.phase_2_preprocessing,
            'glm': self.phase_3_glm_modeling,
            'xgboost': self.phase_4_xgboost_modeling,
            'modeling': lambda: self.phase_3_glm_modeling() and self.phase_4_xgboost_modeling(),
            'comparison': self.phase_5_model_comparison,
            'interpretability': self.phase_6_interpretability,
            'documentation': self.phase_7_documentation
        }

        if phase_name not in phase_methods:
            print(f"Error: Unknown phase '{phase_name}'")
            print(f"Available phases: {', '.join(phase_methods.keys())}")
            return False

        try:
            result = phase_methods[phase_name]()
            return result if result is not None else True
        except Exception as e:
            print(f"Error in phase '{phase_name}': {e}")
            return False

    def run_all(self):
        """Run the complete pipeline"""
        print("Starting Complete ML Pipeline")
        print("="*80)

        phases = [
            'eda',
            'preprocessing',
            'glm',
            'xgboost',
            'comparison',
            'interpretability',
            'documentation'
        ]

        for phase in phases:
            success = self.run_phase(phase)
            if not success:
                print(f"\nPipeline failed at phase: {phase}")
                return False

        print("\n" + "="*80)
        print("COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
        print("="*80)
        print(f"All outputs saved to: {self.output_dir}")
        return True

def main():
    parser = argparse.ArgumentParser(description="Complete ML Pipeline - GLM vs XGBoost")
    parser.add_argument('--all', action='store_true', help='Run the entire pipeline')
    parser.add_argument('--phase',
                       choices=['eda', 'preprocessing', 'glm', 'xgboost', 'modeling', 'comparison', 'interpretability', 'documentation'],
                       help='Run a specific phase')

    args = parser.parse_args()

    pipeline = CompletePipeline()

    if args.all:
        success = pipeline.run_all()
        exit_code = 0 if success else 1
    elif args.phase:
        success = pipeline.run_phase(args.phase)
        exit_code = 0 if success else 1
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python complete_pipeline.py --all")
        print("  python complete_pipeline.py --phase eda")
        print("  python complete_pipeline.py --phase modeling")
        exit_code = 0

    exit(exit_code)

if __name__ == "__main__":
    main()