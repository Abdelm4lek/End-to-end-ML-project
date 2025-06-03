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
        predictor = Predictor(config=predictor_config.model_path)
        predictor.predict_all_stations()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PredictionPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e