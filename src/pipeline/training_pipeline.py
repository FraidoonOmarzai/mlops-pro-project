import sys
from src.logger import logger
from src.exception import CustomException
from src.constants import CONFIG_PATH, MODEL_CONFIG_PATH
from src.utils.common import read_yaml
from src.components.data_ingestion import (DataIngestion,
                                           create_data_ingestion_config
                                           )
from src.components.data_validation import (DataValidation,
                                            create_data_validation_config
                                            )
from src.components.data_preprocessing import (DataPreprocessing,
                                               create_data_preprocessing_config
                                               )
from src.components.model_trainer import (ModelTrainer,
                                          create_model_trainer_config
                                          )
from src.components.model_evaluation import (ModelEvaluation,
                                             create_model_evaluation_config
                                             )


class TrainingPipeline:
    """
    Orchestrates the complete ML training pipeline.
    """

    def __init__(self):
        """Initialize TrainingPipeline."""
        logger.info("="*70)
        logger.info("INITIALIZING TRAINING PIPELINE")
        logger.info("="*70)

        # Create necessary directories
        # config_manager.create_directories()

        self.config_dict = read_yaml(CONFIG_PATH)
        self.model_config_params = read_yaml(MODEL_CONFIG_PATH)

    def start_data_ingestion(self):
        """
        Step 1: Data Ingestion
        Load data and split into train/test sets.
        """
        try:
            logger.info("\n" + "="*70)
            logger.info("STEP 1: DATA INGESTION")
            logger.info("="*70)

            # config_dict = read_yaml(CONFIG_PATH)
            config = create_data_ingestion_config(
                self.config_dict.data_ingestion)
            data_ingestion = DataIngestion(config)
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            logger.info("Data Ingestion completed successfully\n")

            logger.info("Data Ingestion completed successfully\n")
            return train_data_path, test_data_path

        except Exception as e:
            logger.error("Error in Data Ingestion step")
            raise CustomException(e, sys)

    def start_data_validation(self, train_path: str, test_path: str):
        """
        Step 2: Data Validation
        Validate data quality and schema.
        """
        try:
            logger.info("\n" + "="*70)
            logger.info("STEP 2: DATA VALIDATION")
            logger.info("="*70)

            data_validation_config = create_data_validation_config(
                self.config_dict.data_validation)
            data_validation_status = DataValidation(data_validation_config).initiate_data_validation(
                train_path,
                test_path
            )

            if not data_validation_status:
                logger.warning(
                    "Data validation failed! Check validation report.")
            else:
                logger.info("Data Validation completed successfully\n")

            return data_validation_status

        except Exception as e:
            logger.error("Error in Data Validation step")
            raise CustomException(e, sys)

    def start_data_preprocessing(self, train_path: str, test_path: str):
        """
        Step 3: Data Preprocessing
        Clean data and apply transformations.
        """
        try:
            logger.info("\n" + "="*70)
            logger.info("STEP 3: DATA PREPROCESSING")
            logger.info("="*70)

            data_preprocessing_config = create_data_preprocessing_config(
                self.config_dict.data_preprocessing)
            data_preprocessing = DataPreprocessing(data_preprocessing_config)
            X_train, X_test, y_train, y_test, preprocessor_path = data_preprocessing.initiate_data_preprocessing(
                train_path,
                test_path
            )

            logger.info("Data Preprocessing completed successfully\n")

            return X_train, X_test, y_train, y_test, preprocessor_path

        except Exception as e:
            logger.error("Error in Data Preprocessing step")
            raise CustomException(e, sys)

    def start_model_training(self, X_train, X_test, y_train, y_test):
        """
        Step 4: Model Training
        Train multiple ML models with MLflow tracking.
        """
        try:
            logger.info("\n" + "="*70)
            logger.info("STEP 4: MODEL TRAINING")
            logger.info("="*70)

            model_training_config = create_model_trainer_config(
                self.config_dict.model_training,
                self.config_dict.mlflow)

            # Get model parameters
            model_params = {}
            for model_name in self.config_dict.model_training.models:
                model_params[model_name] = self.model_config_params[model_name]

            trained_models = ModelTrainer(
                config=model_training_config,
                model_params=model_params
            ).initiate_model_training(
                X_train,
                X_test,
                y_train,
                y_test
            )

            logger.info("Model Training completed successfully\n")

            return trained_models

        except Exception as e:
            logger.error("Error in Model Training step")
            raise CustomException(e, sys)

    def start_model_evaluation(self, trained_models, X_train, X_test, y_train, y_test):
        """
        Step 5: Model Evaluation
        Evaluate all trained models and select the best one.
        """
        try:
            logger.info("\n" + "="*70)
            logger.info("STEP 5: MODEL EVALUATION")
            logger.info("="*70)

            config = create_model_evaluation_config(
                self.config_dict.model_evaluation)
            evaluation_report = ModelEvaluation(config).initiate_model_evaluation(
                trained_models=trained_models,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test
            )

            logger.info("Model Evaluation completed successfully\n")

            return evaluation_report

        except Exception as e:
            logger.error("Error in Model Evaluation step")
            raise CustomException(e, sys)

    def run_pipeline(self):
        """
        Run the complete training pipeline.
        """
        try:
            logger.info("Starting complete training pipeline...\n")

            # Step 1: Data Ingestion
            train_path, test_path = self.start_data_ingestion()

            # Step 2: Data Validation
            validation_status = self.start_data_validation(
                train_path, test_path)

            if not validation_status:
                logger.warning("Proceeding despite validation warnings...")

            # Step 3: Data Preprocessing
            X_train, X_test, y_train, y_test, preprocessor_path = \
                self.start_data_preprocessing(train_path, test_path)

            # Step 4: Model Training
            trained_models = self.start_model_training(
                X_train, X_test, y_train, y_test)

            # Step 5: Model Evaluation
            evaluation_report = self.start_model_evaluation(
                trained_models, X_train, X_test, y_train, y_test
            )

            # Pipeline completion summary
            logger.info("\n" + "="*70)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*70)
            logger.info(
                f"\nBest Model: {evaluation_report['best_model_name']}")
            logger.info(f"Models trained: {len(trained_models)}")
            logger.info(f"Preprocessor saved at: {preprocessor_path}")
            logger.info("\nCheck MLflow UI for detailed experiment tracking:")
            logger.info("  Run: mlflow ui")
            logger.info("  Open: http://localhost:5000")
            logger.info("="*70 + "\n")

            return {
                'train_path': train_path,
                'test_path': test_path,
                'preprocessor_path': preprocessor_path,
                'trained_models': trained_models,
                'evaluation_report': evaluation_report
            }

        except Exception as e:
            logger.error("Training pipeline failed")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logger.error("Pipeline execution failed")
        raise e
