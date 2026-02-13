#!/usr/bin/env python3
"""
Simple Pipeline Runner
GLM vs XGBoost Modeling Project

Usage:
    python run_pipeline.py --all                    # Run entire pipeline
    python run_pipeline.py --phase eda             # Run only EDA
    python run_pipeline.py --status                # Check pipeline status
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

class SimplePipelineRunner:
    def __init__(self):
        # Pipeline phases in order
        self.phases = [
            ('eda', 'Exploratory Data Analysis', '01_eda/quick_eda.py'),
            ('preprocessing', 'Data Preprocessing', '02_preprocessing/data_preprocessing.py'),
            ('glm', 'GLM Model Development', '03_glm_model/glm_modeling.py'),
            ('xgboost', 'XGBoost Model Development', '04_xgboost_model/xgboost_modeling.py'),
            ('compare', 'Model Comparison', '05_model_comparison/compare_models.py'),
            ('interpret', 'Model Interpretability', '06_interpretability/local_interpretability.py'),
            ('document', 'Final Documentation', '07_final_results/final_documentation.py')
        ]

        # Set up timestamped output directory
        today = datetime.now().strftime('%Y-%m-%d')
        self.output_dir = Path(f'outputs/{today}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        os.environ['OUTPUT_BASE_DIR'] = str(self.output_dir)

        # Create symlink to latest run
        latest_link = Path('outputs/latest')
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        try:
            latest_link.symlink_to(today, target_is_directory=True)
        except (OSError, NotImplementedError):
            pass  # Symlinks not supported

        print(f"Pipeline outputs will be saved to: {self.output_dir}")

    def run_phase(self, phase_name):
        """Run a single phase"""
        # Find the phase
        phase_info = None
        for name, description, script in self.phases:
            if name == phase_name:
                phase_info = (name, description, script)
                break

        if not phase_info:
            print(f"Error: Unknown phase '{phase_name}'")
            return False

        name, description, script = phase_info

        print(f"\nRunning: {description}")
        print(f"Script: {script}")
        print("=" * 60)

        try:
            result = subprocess.run(['uv', 'run', 'python', script], cwd=Path.cwd())
            if result.returncode == 0:
                print(f"\n{description} completed successfully!")
                return True
            else:
                print(f"\n{description} failed with exit code {result.returncode}")
                return False
        except Exception as e:
            print(f"\nError running {description}: {e}")
            return False

    def run_all(self):
        """Run all phases in sequence"""
        print("Starting complete pipeline execution")
        print("=" * 60)

        failed_phases = []

        for phase_name, description, script in self.phases:
            success = self.run_phase(phase_name)
            if not success:
                failed_phases.append(phase_name)
                print(f"\nPipeline stopped at {phase_name} due to failure.")
                break

        print("\nPipeline Summary:")
        if not failed_phases:
            print("All phases completed successfully!")
        else:
            print(f"Pipeline failed at phase: {failed_phases[0]}")

        return len(failed_phases) == 0

    def show_status(self):
        """Show pipeline status"""
        print("Pipeline Status")
        print("=" * 60)

        for phase_name, description, script in self.phases:
            print(f"{description:<40} [READY]")
            print(f"  Script: {script}")

        print(f"\nOutputs will be saved to: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="GLM vs XGBoost Pipeline Runner")
    parser.add_argument('--all', action='store_true', help='Run the entire pipeline')
    parser.add_argument('--phase', choices=['eda', 'preprocessing', 'glm', 'xgboost', 'compare', 'interpret', 'document'], help='Run a specific phase')
    parser.add_argument('--status', action='store_true', help='Show pipeline status')

    args = parser.parse_args()
    runner = SimplePipelineRunner()

    if args.status:
        runner.show_status()
    elif args.all:
        success = runner.run_all()
        sys.exit(0 if success else 1)
    elif args.phase:
        success = runner.run_phase(args.phase)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python run_pipeline.py --status")
        print("  python run_pipeline.py --all")
        print("  python run_pipeline.py --phase eda")

if __name__ == "__main__":
    main()