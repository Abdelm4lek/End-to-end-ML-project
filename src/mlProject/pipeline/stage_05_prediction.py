from mlProject.config.configuration import ConfigurationManager
from mlProject.components.prediction import Predictor
from mlProject import logger



STAGE_NAME = "Prediction stage"

class PredictionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        predictor_config = config.get_prediction_config()
        predictor = Predictor(model_path=predictor_config.model_path)
        predictions = predictor.predict_all_stations()

        if predictions:
            predictor.save_predictions_to_csv(predictions, str(predictor_config.predictions_path))
            logger.info(f"Predictions saved successfully for {len(predictions)} stations")
        else:
            logger.warning("No predictions were made - nothing to save")
        
        

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        obj = PredictionPipeline()
        obj.main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\n{'='*80}")
    except Exception as e:
        logger.exception(e)
        raise e