from mlProject.constants import *
from mlProject.utils.common import read_yaml, create_dirs
from mlProject.entity.config_entity import (DataIngestionConfig,
                                            DataValidationConfig,
                                            DataTransformationConfig,
                                            ModelTrainerConfig,
                                            PredictionConfig,
                                            ProductionRetrainingConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_dirs([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        mode = config.mode
        
        if mode == "local":
            local_config = config.local
            create_dirs([local_config.root_dir])
            
            data_ingestion_config = DataIngestionConfig(
                root_dir=local_config.root_dir,
                unzip_data_path=local_config.unzip_data_path,
                mode=mode,
                source_URL=local_config.source_url,
                local_data_file=local_config.local_data_file,
                unzip_dir=local_config.unzip_dir
            )
        else:  # production mode
            prod_config = config.production
            create_dirs([prod_config.root_dir])
            
            data_ingestion_config = DataIngestionConfig(
                root_dir=prod_config.root_dir,
                unzip_data_path=prod_config.unzip_data_path,
                mode=mode,
                training_window_days=prod_config.training_window_days,
                min_data_points=prod_config.min_data_points
            )

        return data_ingestion_config
    

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_dirs([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir = config.unzip_data_dir,
            all_schema=schema,
        )

        return data_validation_config
    

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_dirs([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path
        )

        return data_transformation_config



    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.LightGBM
        target =  self.schema.TARGET_COLUMN
        # Get MLflow URI from model_evaluation config
        mlflow_uri = self.config.model_evaluation.mlflow_uri

        create_dirs([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path = config.train_data_path,
            test_data_path = config.test_data_path,
            model_name = config.model_name,
            objective = params.objective,
            metric = params.metric,
            boosting_type = params.boosting_type,
            num_leaves = params.num_leaves,
            learning_rate = params.learning_rate,
            feature_fraction = params.feature_fraction,
            n_estimators = params.n_estimators,
            target_column = target.name,
            mlflow_uri = config.mlflow_uri
        )

        return model_trainer_config

    def get_prediction_config(self) -> PredictionConfig:
        config = self.config.prediction

        create_dirs([config.root_dir])

        prediction_config = PredictionConfig(
            root_dir=config.root_dir,
            model_path=config.model_path,
            prediction_window_hours=config.prediction_window_hours,
            min_data_points=config.min_data_points,
            prediction_interval_minutes=config.prediction_interval_minutes,
            predictions_path=config.predictions_path
        )

        return prediction_config
    
    def get_production_retraining_config(self) -> ProductionRetrainingConfig:
        config = self.config.production_retraining

        production_retraining_config = ProductionRetrainingConfig(
            data_source=config.data_source,
            training_window_days=config.training_window_days,
            min_data_points=config.min_data_points,
            retrain_frequency=config.retrain_frequency,
            auto_retrain=config.auto_retrain,
            min_improvement_threshold=config.min_improvement_threshold,
            experiment_name=config.experiment_name,
            mlflow_uri=config.mlflow_uri,
            model_name=config.model_name,
            model_stage=config.model_stage,
            fallback_to_artifacts=config.fallback_to_artifacts,
            artifacts_backup_enabled=config.artifacts_backup_enabled
        )

        return production_retraining_config