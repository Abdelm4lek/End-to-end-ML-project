import time
import gc
import sys
from mlProject import logger
from mlProject.pipeline.data_ingestion_stage import DataIngestionPipeline
from mlProject.pipeline.data_validation_stage import DataValidationPipeline
from mlProject.pipeline.data_transformation_stage import DataTransformationPipeline
from mlProject.pipeline.model_training_stage import ModelTrainerPipeline
from mlProject.pipeline.model_evaluation_stage import ModelEvaluationPipeline
from mlProject.pipeline.model_deployment_stage import ModelDeploymentPipeline
from mlProject.pipeline.prediction_stage import PredictionPipeline
from multiprocessing import Process

if __name__ == "__main__":
    start_time = time.time()
    logger.info(f"Run started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

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

    # Run deployment as subprocess to avoid MLflow/Hopsworks resource conflicts
    # between model evaluation and deployment stages
    STAGE_NAME = "Model Deployment stage"
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        p = Process(target=ModelDeploymentPipeline().main)
        p.start()
        p.join()
        if p.exitcode != 0:
            logger.error("Model Deployment stage failed")
            exit(p.exitcode)
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\n{'='*100}")
    except Exception as e:
        logger.exception(e)
        raise e

    # Run prediction as subprocess to avoid Hopsworks resource conflicts
    # from previous stages (data ingestion, model deployment)
    STAGE_NAME = "Prediction stage"
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        p = Process(target=PredictionPipeline().main)
        p.start()
        p.join()
        if p.exitcode != 0:
            logger.error("Prediction stage failed")
            exit(p.exitcode)
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\n{'='*100}")
    except Exception as e:
        logger.exception(e)
        raise e

    # Final resource cleanup
    logger.info("Final cleanup and exit.")
    gc.collect()
    elapsed = time.time() - start_time
    logger.info(f"Run completed in {elapsed:.2f} seconds.")
    sys.exit(0)