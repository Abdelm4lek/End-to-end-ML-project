import os
import mlflow
import mlflow.lightgbm
from mlflow.tracking import MlflowClient
from typing import Dict, Optional, Any
import joblib
import json
import tempfile
from dataclasses import dataclass
from src.mlProject import logger
from src.mlProject.hopsworks.model_registry import HopsworksModelRegistry
from src.mlProject.hopsworks.config import HopsworksConfig, MLflowBridgeConfig
from src.mlProject.utils.common import save_json
from pathlib import Path 


class MLflowBridge:
    """
    Simple bridge between MLflow and Hopsworks Model Registry.
    
    Automatically deploys models from MLflow to Hopsworks when they
    perform better than defined thresholds or the current deployed model.
    """
    def __init__(self, config: MLflowBridgeConfig):
        self.config = config
        self.mlflow_client = None
        self.hopsworks_registry = None
        self._setup_connections()
        
    
    def _setup_connections(self):
        """Setup MLflow and Hopsworks connections."""
        try:
            # Setup MLflow
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            self.mlflow_client = MlflowClient(tracking_uri=self.config.mlflow_tracking_uri) 
            logger.info(f"MLflow client connected to: {self.config.mlflow_tracking_uri}")
            
            # Setup Hopsworks
            hopsworks_config = self.config.hopsworks_config
            self.hopsworks_registry = HopsworksModelRegistry(hopsworks_config)
            logger.info("Hopsworks registry connected successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup connections: {str(e)}")
            raise e
    
    def get_latest_mlflow_model(self, experiment_name: str) -> Optional[Dict]:
        """Get the latest model from MLflow experiment."""
        try:
            # Get experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                logger.error(f"Experiment '{experiment_name}' not found")
                return None
            
            # Get the latest run from the experiment
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if runs.empty:
                logger.warning(f"No runs found in experiment '{experiment_name}'")
                return None
            
            latest_run = runs.iloc[0]
            
            # Extract metrics and run info
            model_info = {
                "run_id": latest_run["run_id"],
                "metrics": {
                    "rmse": latest_run.get("metrics.rmse", float('inf')),
                    "mae": latest_run.get("metrics.mae", float('inf')),
                    "r2": latest_run.get("metrics.r2", 0.0)
                },
                "start_time": latest_run["start_time"],
                "status": latest_run["status"]
            }
            
            logger.info(f"Retrieved latest model from MLflow: {model_info}")
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to get latest MLflow model: {str(e)}")
            return None
    
    def download_model_from_mlflow(self, run_id: str) -> Optional[str]:
        """Download model from MLflow run to temporary location."""
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            model_path = os.path.join(temp_dir, "model.joblib")
            
            # Download model artifact
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.lightgbm.load_model(model_uri)
            
            # Save to temporary location
            joblib.dump(model, model_path)
            logger.info(f"Downloaded model from run {run_id} to {model_path}")
            
            return model_path
            
        except Exception as e:
            logger.error(f"Failed to download model from run {run_id}: {str(e)}")
            return None
    
    def meets_performance_threshold(self, metrics: Dict[str, float]) -> bool:
        """Check if model metrics meet deployment thresholds."""
        try:
            for metric_name, threshold_value in self.config.performance_threshold.items():
                if metric_name not in metrics:
                    logger.warning(f"Metric {metric_name} not found in model metrics")
                    return False
                
                metric_value = metrics[metric_name]
                
                # For error metrics (rmse, mae), lower is better
                if metric_name.lower() in ["rmse", "mae", "mse"]:
                    if metric_value > threshold_value:
                        logger.info(f"Model failed threshold: {metric_name}={metric_value} > {threshold_value}")
                        return False
                # For performance metrics (r2, accuracy), higher is better
                else:
                    if metric_value < threshold_value:
                        logger.info(f"Model failed threshold: {metric_name}={metric_value} < {threshold_value}")
                        return False
            
            logger.info("Model meets all performance thresholds")
            return True
            
        except Exception as e:
            logger.error(f"Error checking thresholds: {str(e)}")
            return False
    
    def is_better_than_current_model(self, new_metrics: Dict[str, float]) -> bool:
        """Check if new model performs better than current Hopsworks model."""
        try:
            # Get current model metrics from Hopsworks
            current_metrics = self.hopsworks_registry.get_model_metrics()
            
            if not current_metrics:
                logger.info("No current model in Hopsworks, new model will be deployed")
                return True
            
            # Compare key metrics
            for metric_name in ["rmse", "mae", "r2"]:
                if metric_name not in new_metrics or metric_name not in current_metrics:
                    continue
                
                new_value = new_metrics[metric_name]
                current_value = current_metrics[metric_name]
                
                # For error metrics, lower is better
                if metric_name.lower() in ["rmse", "mae", "mse"]:
                    if new_value >= current_value:
                        logger.info(f"New model not better: {metric_name} {new_value} >= {current_value}")
                        return False
                # For performance metrics, higher is better
                else:
                    if new_value <= current_value:
                        logger.info(f"New model not better: {metric_name} {new_value} <= {current_value}")
                        return False
            
            logger.info("New model performs better than current model")
            return True
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            return False
    
    def deploy_to_hopsworks(self, model_path: str, metrics: Dict[str, float], description: str = None) -> bool:
        """Deploy model to Hopsworks registry."""
        try:
            # Load model
            model = joblib.load(model_path)
            
            # Create deployment description
            if not description:
                description = f"Auto-deployed from MLflow - RMSE: {metrics.get('rmse', 'N/A')}, MAE: {metrics.get('mae', 'N/A')}, R2: {metrics.get('r2', 'N/A')}"
            
            # Deploy to Hopsworks
            self.hopsworks_registry.save_model(
                model=model,
                metrics=metrics,
                description=description
            )
            
            logger.info(f"Successfully deployed model to Hopsworks: {description}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy model to Hopsworks: {str(e)}")
            return False
    
    def log_deployment_result(self, success: bool, model_info: Dict, reason: str = None):
        """Log deployment result to file."""
        try:
            log_entry = {
                "timestamp": str(model_info.get("start_time", "unknown")),
                "run_id": model_info.get("run_id", "unknown"),
                "metrics": model_info.get("metrics", {}),
                "deployment_success": success,
                "reason": reason or ("Deployment successful" if success else "Deployment failed")
            }
            
            # Read existing log or create new
            try:
                with open(self.config.deployment_log_file, 'r') as f:
                    log_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                log_data = {"deployments": []}
            
            log_data["deployments"].append(log_entry)
            
            # Save updated log
            save_json(path=Path(self.config.deployment_log_file), data=log_data)
            logger.info(f"Deployment result logged to {self.config.deployment_log_file}")
            
        except Exception as e:
            logger.error(f"Failed to log deployment result: {str(e)}")
    
    def sync_and_deploy(self) -> bool:
        """
        Main method: Get latest MLflow model, evaluate, and deploy to Hopsworks if better.
        
        Returns:
            True if deployment was successful, False otherwise
        """
        try:
            logger.info("Starting MLflow to Hopsworks sync and deployment...")
            
            # 1. Get latest model from MLflow
            model_info = self.get_latest_mlflow_model(self.config.evaluation_experiment_name)
            if not model_info:
                logger.error("No model found in MLflow")
                return False
            
            # 2. Check if model meets performance thresholds
            if not self.meets_performance_threshold(model_info["metrics"]):
                reason = "Model does not meet performance thresholds"
                logger.info(f"{reason}, skipping deployment")
                self.log_deployment_result(False, model_info, reason)
                return False
            
            # 3. Check if model is better than current deployed model
            if not self.is_better_than_current_model(model_info["metrics"]):
                reason = "Model is not better than current deployed model"
                logger.info(f"{reason}, skipping deployment")
                self.log_deployment_result(False, model_info, reason)
                return False
            
            # 4. Download model from MLflow
            model_path = self.download_model_from_mlflow(model_info["run_id"])
            if not model_path:
                reason = "Failed to download model from MLflow"
                logger.error(reason)
                self.log_deployment_result(False, model_info, reason)
                return False
            
            # 5. Deploy to Hopsworks (if available)
            success = self.deploy_to_hopsworks(
                model_path=model_path,
                metrics=model_info["metrics"],
                description=f"Auto-deployed from MLflow run {model_info['run_id']}"
            )
            
            # 6. Cleanup temporary files
            try:
                os.remove(model_path)
                os.rmdir(os.path.dirname(model_path))
            except:
                pass
            
            # 7. Log deployment result
            if success:
                reason = "Successfully deployed to Hopsworks"
                logger.info("MLflow to Hopsworks deployment completed successfully!")
            else:
                reason = "Model meets criteria but Hopsworks deployment failed"
                logger.warning(reason)
            
            self.log_deployment_result(success, model_info, reason)
            return success
            
        except Exception as e:
            logger.error(f"Error in sync_and_deploy: {str(e)}")
            return False 
        


from src.mlProject.hopsworks.config import HopsworksConfig

if __name__ == "__main__":
    # Configuration
    config = MLflowBridgeConfig()

    # Create bridge and deploy
    bridge = MLflowBridge(config)
    success = bridge.sync_and_deploy()  # Returns True if deployed