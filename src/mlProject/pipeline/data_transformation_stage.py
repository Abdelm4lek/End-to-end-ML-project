from mlProject.config.configuration import ConfigurationManager
from mlProject.components.data_transformation import DataTransformation
from mlProject import logger
from pathlib import Path




STAGE_NAME = "Data Transformation stage"

class DataTransformationPipeline:
    def __init__(self):
        pass


    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(config=data_transformation_config)
                peprocessed_df = data_transformation.preprocess_data()
                lags_df = data_transformation.create_lagged_features(peprocessed_df)
                data_transformation.split_train_test(lags_df)

                del peprocessed_df, lags_df
                logger.info("Data transformation stage completed successfully")

            else:
                raise Exception("You data schema is not valid")

        except Exception as e:
            print(e)
            


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        obj = DataTransformationPipeline()
        obj.main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\n{'='*80}")
    except Exception as e:
        logger.exception(e)
        raise e