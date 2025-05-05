import os
from mlProject.entity.config_entity import ModelTrainerConfig
from mlProject import logger
import polars as pl
import lightgbm as lgb
import joblib


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
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
        # It automatically detects 'category' dtype columns
        lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False) # Keep raw data if needed later
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
            'verbose': -1,  # avoids surpressing training process messages
            'n_estimators': self.config.n_estimators,     
            'n_jobs': -1   # Use all available CPU cores
        }

        # Train the model
        lgbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=1000, # Max rounds
                        valid_sets=[lgb_train, lgb_eval],
                        valid_names=['train', 'eval'],
                        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(period=50)])
        
        joblib.dump(lgbm, os.path.join(self.config.root_dir, self.config.model_name))
        logger.info(f"Model saved to {self.config.root_dir}/{self.config.model_name}")