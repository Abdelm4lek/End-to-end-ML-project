from mlProject import logger
from mlProject.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from mlProject.pipeline.stage_02_data_validation import DataValidationPipeline
from mlProject.pipeline.stage_03_data_transformation import DataTransformationPipeline
from mlProject.pipeline.stage_04_model_trainer import ModelTrainerPipeline
from mlProject.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline
from mlProject.pipeline.stage_06_prediction import PredictionPipeline


STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> {STAGE_NAME} started <<<<<<") 
   data_ingestion_pipeline = DataIngestionPipeline()
   data_ingestion_pipeline.main()
   logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\n{'='*100}")
except Exception as e:
        logger.exception(e)
        raise e




STAGE_NAME = "Data Validation stage"
try:
   logger.info(f">>>>>> {STAGE_NAME} started <<<<<<") 
   data_validation_pipeline = DataValidationPipeline()
   data_validation_pipeline.main()
   logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\n{'='*100}")
except Exception as e:
        logger.exception(e)
        raise e



STAGE_NAME = "Data Transformation stage"
try:
   logger.info(f">>>>>> {STAGE_NAME} started <<<<<<") 
   data_transformation_pipeline = DataTransformationPipeline()
   data_transformation_pipeline.main()
   logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\n{'='*100}")
except Exception as e:
        logger.exception(e)
        raise e




STAGE_NAME = "Model Trainer stage"
try:
   logger.info(f">>>>>> {STAGE_NAME} started <<<<<<") 
   model_trainer_pipeline = ModelTrainerPipeline()
   model_trainer_pipeline.main()
   logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\n{'='*100}")
except Exception as e:
        logger.exception(e)
        raise e



STAGE_NAME = "Model Evaluation stage"
try:
   logger.info(f">>>>>> {STAGE_NAME} started <<<<<<") 
   model_evaluation_pipeline = ModelEvaluationPipeline()
   model_evaluation_pipeline.main()
   logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\n{'='*100}")
except Exception as e:
        logger.exception(e)
        raise e



STAGE_NAME = "Prediction stage"
try:
   logger.info(f">>>>>> {STAGE_NAME} started <<<<<<") 
   prediction_pipeline = PredictionPipeline()
   prediction_pipeline.main()
   logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\n{'='*100}")
except Exception as e:
        logger.exception(e)
        raise e