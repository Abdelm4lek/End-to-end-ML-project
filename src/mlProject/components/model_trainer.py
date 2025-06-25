import os
from mlProject.entity.config_entity import ModelTrainerConfig
from mlProject import logger
import polars as pl
import lightgbm as lgb
import joblib
import mlflow
import mlflow.lightgbm
from datetime import datetime


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def setup_mlflow(self):
        """Setup MLflow tracking for the experiment."""
        try:
            # Set tracking URI from config
            mlflow_uri = getattr(self.config, 'mlflow_uri', None)
            if mlflow_uri:
                mlflow.set_tracking_uri(mlflow_uri)
                logger.info(f"MLflow tracking URI set to: {mlflow_uri}")
            else:
                logger.warning("No MLflow URI found in config, using local tracking")
            
            experiment_name = "velib_demand_prediction"
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(experiment_name)
                    logger.info(f"Created new MLflow experiment: {experiment_name}")
                else:
                    experiment_id = experiment.experiment_id
                    logger.info(f"Using existing MLflow experiment: {experiment_name}")
                
                mlflow.set_experiment(experiment_name)
                return True
                
            except Exception as e:
                logger.error(f"Error setting up MLflow experiment: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting up MLflow: {str(e)}")
            return False

    def train(self):
        # Setup MLflow tracking
        mlflow_enabled = self.setup_mlflow()
        
        if mlflow_enabled:
            # Start MLflow run
            with mlflow.start_run() as run:
                logger.info(f"Started MLflow run: {run.info.run_id}")
                self._train_with_mlflow()
        else:
            logger.warning("MLflow not available, training without tracking")
            self._train_without_mlflow()
    
    def _train_with_mlflow(self):
        """Train model with MLflow tracking."""
        # read the train and test datasets using polars since it's faster
        train_data = pl.read_csv(self.config.train_data_path)
        test_data = pl.read_csv(self.config.test_data_path)

        # convert to pandas dataframe since the model handles them better
        train_data = train_data.to_pandas() 
        test_data = test_data.to_pandas()
        print(train_data['date'].dtype)

        X_train = train_data.drop(['date', self.config.target_column], axis=1)
        X_test = test_data.drop(['date', self.config.target_column], axis=1)
        y_train = train_data[[self.config.target_column]]
        y_test = test_data[[self.config.target_column]]

        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        # Log dataset information to MLflow
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("target_column", self.config.target_column)

        # Create LightGBM Dataset objects
        lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, free_raw_data=False)

        # Define model parameters
        params = {
            'objective': self.config.objective,  
            'metric': self.config.metric,              
            'boosting_type': self.config.boosting_type,
            'num_leaves': self.config.num_leaves,
            'learning_rate': self.config.learning_rate,
            'feature_fraction': self.config.feature_fraction,
            'random_state': 42,
            'verbose': -1,
            'n_estimators': self.config.n_estimators,     
            'n_jobs': -1
        }

        # Log hyperparameters to MLflow
        mlflow.log_params(params)
        mlflow.log_param("max_rounds", 1000)
        mlflow.log_param("early_stopping_rounds", 10)

        # Train the model
        lgbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=1000,
                        valid_sets=[lgb_train, lgb_eval],
                        valid_names=['train', 'eval'],
                        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(period=50)])
        
        # Log model to MLflow
        mlflow.lightgbm.log_model(lgbm, "model")
        
        # Save model locally as well
        model_path = os.path.join(self.config.root_dir, self.config.model_name)
        joblib.dump(lgbm, model_path)
        
        # Log model artifact
        mlflow.log_artifact(model_path, "local_model")
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Model logged to MLflow run: {mlflow.active_run().info.run_id}")
    
    def _train_without_mlflow(self):
        """Fallback training method without MLflow (original logic)."""
        # read the train and test datasets using polars since it's faster
        train_data = pl.read_csv(self.config.train_data_path)
        test_data = pl.read_csv(self.config.test_data_path)

        # convert to pandas dataframe since the model handles them better
        train_data = train_data.to_pandas() 
        test_data = test_data.to_pandas()
        print(train_data['date'].dtype)

        X_train = train_data.drop(['date', self.config.target_column], axis=1)
        X_test = test_data.drop(['date', self.config.target_column], axis=1)
        y_train = train_data[[self.config.target_column]]
        y_test = test_data[[self.config.target_column]]

        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        # Create LightGBM Dataset objects
        lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, free_raw_data=False)

        # Define model parameters
        params = {
            'objective': self.config.objective,  
            'metric': self.config.metric,              
            'boosting_type': self.config.boosting_type,
            'num_leaves': self.config.num_leaves,
            'learning_rate': self.config.learning_rate,
            'feature_fraction': self.config.feature_fraction,
            'random_state': 42,
            'verbose': -1,
            'n_estimators': self.config.n_estimators,     
            'n_jobs': -1
        }

        # Train the model
        lgbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=1000,
                        valid_sets=[lgb_train, lgb_eval],
                        valid_names=['train', 'eval'],
                        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(period=50)])
        
        joblib.dump(lgbm, os.path.join(self.config.root_dir, self.config.model_name))
        logger.info(f"Model saved to {self.config.root_dir}/{self.config.model_name}")