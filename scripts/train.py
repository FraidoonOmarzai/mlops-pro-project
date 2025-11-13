"""
Training script to run the complete ML pipeline.
Usage: python scripts/train.py
"""

import os
import sys
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logger import logger
from src.exception import CustomException
from src.pipeline.training_pipeline import TrainingPipeline


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ML models for churn prediction')
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file (default: configs/config.yaml)'
    )
    
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip data validation step'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['logistic_regression', 'random_forest', 'xgboost', 'lightgbm'],
        help='Specific models to train (default: all)'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='MLflow experiment name'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def check_prerequisites():
    """Check if all prerequisites are met before training."""
    logger.info("Checking prerequisites...")
    
    # Check if data file exists
    data_file = "data/raw/churn_data.csv"
    if not os.path.exists(data_file):
        logger.error(f"Dataset not found at {data_file}")
        logger.info("Please run: python scripts/download_data.py")
        return False
    
    # Check if config files exist
    if not os.path.exists("configs/config.yaml"):
        logger.error("Configuration file not found: configs/config.yaml")
        return False
    
    if not os.path.exists("configs/model_config.yaml"):
        logger.error("Model configuration file not found: configs/model_config.yaml")
        return False
    
    # Check directories
    required_dirs = ['logs', 'artifacts/models', 'artifacts/preprocessors', 'artifacts/metrics']
    for directory in required_dirs:
        if not os.path.exists(directory):
            logger.info(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)
    
    logger.info("✓ All prerequisites met")
    return True


def print_banner():
    """Print training banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║           CUSTOMER CHURN PREDICTION - ML TRAINING                ║
    ║                                                                  ║
    ║              MLOps Pipeline - Model Training Phase               ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_training_config(args):
    """Print training configuration."""
    logger.info("\n" + "="*70)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*70)
    logger.info(f"Configuration file: {args.config}")
    logger.info(f"Skip validation: {args.skip_validation}")
    logger.info(f"Models to train: {args.models if args.models else 'All models'}")
    logger.info(f"Experiment name: {args.experiment_name if args.experiment_name else 'Default'}")
    logger.info(f"Verbose mode: {args.verbose}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70 + "\n")


def print_results_summary(results):
    """Print training results summary."""
    logger.info("\n" + "="*70)
    logger.info("TRAINING RESULTS SUMMARY")
    logger.info("="*70)
    
    # Model information
    logger.info(f"\n📊 Models Trained: {len(results['trained_models'])}")
    for model_name in results['trained_models'].keys():
        logger.info(f"   ✓ {model_name}")
    
    # Best model
    best_model = results['evaluation_report']['best_model_name']
    logger.info(f"\n🏆 Best Model: {best_model}")
    
    # Evaluation metrics
    best_model_metrics = results['evaluation_report']['individual_evaluations'][best_model]['test_metrics']
    logger.info(f"\n📈 Performance Metrics (Test Set):")
    logger.info(f"   • Accuracy:  {best_model_metrics['accuracy']:.4f}")
    logger.info(f"   • Precision: {best_model_metrics['precision']:.4f}")
    logger.info(f"   • Recall:    {best_model_metrics['recall']:.4f}")
    logger.info(f"   • F1-Score:  {best_model_metrics['f1_score']:.4f}")
    if 'roc_auc' in best_model_metrics:
        logger.info(f"   • ROC-AUC:   {best_model_metrics['roc_auc']:.4f}")
    
    # Artifacts location
    logger.info(f"\n📁 Artifacts Saved:")
    logger.info(f"   • Models:        artifacts/models/")
    logger.info(f"   • Preprocessor:  {results['preprocessor_path']}")
    logger.info(f"   • Metrics:       artifacts/metrics/evaluation_report.json")
    logger.info(f"   • MLflow Runs:   mlruns/")
    
    # Next steps
    logger.info(f"\n🚀 Next Steps:")
    logger.info(f"   1. View MLflow UI:    mlflow ui")
    logger.info(f"   2. Test predictions:  python test_api.py")
    logger.info(f"   3. Start API:         python run_api.py")
    logger.info(f"   4. Start Dashboard:   python run_streamlit.py")
    
    logger.info("="*70 + "\n")


def save_training_summary(results, start_time, end_time):
    """Save training summary to file."""
    try:
        summary_file = f"logs/training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(summary_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("TRAINING SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Start Time: {start_time}\n")
            f.write(f"End Time: {end_time}\n")
            f.write(f"Duration: {(end_time - start_time).total_seconds():.2f} seconds\n\n")
            
            f.write(f"Best Model: {results['evaluation_report']['best_model_name']}\n\n")
            
            f.write("Models Trained:\n")
            for model_name in results['trained_models'].keys():
                f.write(f"  - {model_name}\n")
            
            f.write("\nArtifacts:\n")
            f.write(f"  - Preprocessor: {results['preprocessor_path']}\n")
            f.write(f"  - Models: artifacts/models/\n")
            f.write(f"  - Metrics: artifacts/metrics/\n")
            
        logger.info(f"Training summary saved to: {summary_file}")
        
    except Exception as e:
        logger.warning(f"Could not save training summary: {str(e)}")


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Print banner
    print_banner()
    
    # Record start time
    start_time = datetime.now()
    
    try:
        # Check prerequisites
        if not check_prerequisites():
            logger.error("Prerequisites check failed. Exiting.")
            sys.exit(1)
        
        # Print configuration
        print_training_config(args)
        
        # Initialize pipeline
        logger.info("Initializing training pipeline...")
        pipeline = TrainingPipeline()
        
        # Run pipeline
        logger.info("Starting training pipeline execution...\n")
        results = pipeline.run_pipeline()
        
        # Record end time
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Print results
        print_results_summary(results)
        
        # Save summary
        save_training_summary(results, start_time, end_time)
        
        # Final message
        logger.info(f"✓ Training completed successfully in {duration:.2f} seconds")
        logger.info(f"✓ Total duration: {duration//60:.0f} minutes {duration%60:.0f} seconds")
        
        # Exit successfully
        sys.exit(0)
        
    except CustomException as e:
        logger.error(f"Training failed with custom exception: {str(e)}")
        sys.exit(1)
        
    except KeyboardInterrupt:
        logger.warning("\n\nTraining interrupted by user")
        logger.info("Cleaning up...")
        sys.exit(130)
        
    except Exception as e:
        logger.error(f"Training failed with unexpected error: {str(e)}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        sys.exit(1)