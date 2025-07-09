import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import dotenv
from mlProject.config.configuration import ConfigurationManager
from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.utils.common import save_json
from mlProject import logger



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        """Calculate evaluation metrics for regression model."""
        try:
            rmse = np.sqrt(mean_squared_error(actual, pred))
            mae = mean_absolute_error(actual, pred)
            r2 = r2_score(actual, pred)
            return rmse, mae, r2
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise e

    def setup_mlflow(self):
        """Setup MLflow tracking and registry URIs."""
        try:
            # Load environment variables if available
            dotenv.load_dotenv("DB_credentials.env")
            
            # Set tracking URI
            mlflow_uri = self.config.mlflow_uri
            if mlflow_uri:
                mlflow.set_tracking_uri(mlflow_uri)
                mlflow.set_registry_uri(mlflow_uri)
                logger.info(f"MLflow URI set to: {mlflow_uri}")
            else:
                logger.warning("No MLflow URI found in config, using local tracking")
            
            # Set up experiment for model evaluation
            experiment_name = "velib_demand_model_evaluation"
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is not None:
                    logger.info(f"Using existing MLflow experiment: {experiment_name}")
                else:
                    experiment_id = mlflow.create_experiment(experiment_name)
                    logger.info(f"Created new MLflow experiment: {experiment_name} with ID: {experiment_id}")
                
                mlflow.set_experiment(experiment_name)
                logger.info(f"MLflow experiment set to: {experiment_name}")
                return True
            except mlflow.exceptions.MlflowException as e:
                logger.error(f"Error accessing or creating MLflow experiment: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting up MLflow: {str(e)}")
            return False

    def load_test_data(self):
        """Load and prepare test data."""
        try:
            # Check if test data file exists
            if not os.path.exists(self.config.test_data_path):
                raise FileNotFoundError(f"Test data file not found: {self.config.test_data_path}")
            
            test_data = pd.read_csv(self.config.test_data_path)
            logger.info(f"Loaded test data with shape: {test_data.shape}")
            
            # Handle date column if present (similar to model trainer)
            if 'date' in test_data.columns:
                test_x = test_data.drop(['date', self.config.target_column], axis=1)
            else:
                test_x = test_data.drop([self.config.target_column], axis=1)
                
            test_y = test_data[[self.config.target_column]]
            
            logger.info(f"Test features shape: {test_x.shape}, Test target shape: {test_y.shape}")
            return test_x, test_y
            
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            raise e

    def load_model(self):
        """Load the trained model."""
        try:
            if not os.path.exists(self.config.model_path):
                raise FileNotFoundError(f"Model file not found: {self.config.model_path}")
            
            model = joblib.load(self.config.model_path)
            logger.info(f"Successfully loaded model from: {self.config.model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e

    def save_metrics_locally(self, metrics):
        """Save evaluation metrics to local JSON file."""
        try:
            save_json(path=Path(self.config.metric_file_name), data=metrics)
            logger.info(f"Metrics saved locally to: {self.config.metric_file_name}")
        except Exception as e:
            logger.error(f"Error saving metrics locally: {str(e)}")
            raise e

    def evaluate(self):
        """Evaluate model and log metrics and artifacts to MLflow."""
        try:
            # Setup MLflow
            mlflow_enabled = self.setup_mlflow()
            
            # Load test data and model
            test_x, test_y = self.load_test_data()
            model = self.load_model()
            
            # Check URL type for model registration decision
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            if mlflow_enabled:
                # Start MLflow run for evaluation
                with mlflow.start_run(run_name="model_evaluation") as run:
                    logger.info(f"Started MLflow evaluation run: {run.info.run_id}")
                    
                    # Make predictions
                    predicted_y = model.predict(test_x)
                    
                    # Calculate metrics
                    rmse, mae, r2 = self.eval_metrics(test_y, predicted_y)
                    
                    # Create metrics dictionary
                    scores = {"rmse": rmse, "mae": mae, "r2": r2}
                    
                    # Save metrics locally
                    self.save_metrics_locally(scores)
                    
                    # Log parameters to MLflow
                    mlflow.log_params(self.config.all_params)
                    
                    # Log evaluation dataset info
                    mlflow.log_param("eval_samples", len(test_x))
                    mlflow.log_param("eval_features", test_x.shape[1])
                    
                    # Log metrics to MLflow
                    mlflow.log_metric("rmse", rmse)
                    mlflow.log_metric("mae", mae) 
                    mlflow.log_metric("r2", r2)
                    
                    # Log metrics file as artifact
                    mlflow.log_artifact(str(self.config.metric_file_name), "evaluation_metrics")
                    
                    # Register model based on tracking store type
                    if tracking_url_type_store != "file":
                        # Remote registry - register with name
                        try:
                            mlflow.lightgbm.log_model(
                                model, 
                                "model", 
                                registered_model_name="VelibDemandLGBMRegressor",
                                input_example=test_x.head(5),
                                signature=mlflow.models.infer_signature(test_x, predicted_y[:100])
                            )
                            logger.info("Model registered in remote MLflow registry")
                        except Exception as e:
                            logger.warning(f"Failed to register model: {str(e)}, logging without registration")
                            mlflow.lightgbm.log_model(model, "model")
                    else:
                        # Local tracking - just log the model
                        mlflow.lightgbm.log_model(
                            model, 
                            "model",
                            input_example=test_x.head(5),
                            signature=mlflow.models.infer_signature(test_x, predicted_y[:100])
                        )
                        logger.info("Model logged to local MLflow tracking")
                    
                    logger.info(f"Model evaluation completed. RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
                    
            else:
                # Fallback: evaluate without MLflow
                logger.warning("MLflow not available, performing evaluation without tracking")
                predicted_y = model.predict(test_x)
                rmse, mae, r2 = self.eval_metrics(test_y, predicted_y)
                scores = {"rmse": rmse, "mae": mae, "r2": r2}
                self.save_metrics_locally(scores)
                logger.info(f"Evaluation completed (local only). RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
                
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            raise e


if __name__ == "__main__":
    try:
    # Initialize configuration
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()

        # Create and run evaluation
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.evaluate()
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        raise e