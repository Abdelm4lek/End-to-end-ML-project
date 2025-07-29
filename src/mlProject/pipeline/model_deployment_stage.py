from mlProject.config.configuration import ConfigurationManager
from mlProject.hopsworks.mlflow_bridge import MLflowBridge
from mlProject import logger



STAGE_NAME = "Model Deployment stage"

class ModelDeploymentPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_deployment_config = config.get_model_deployment_config()
        # Initialize the bridge
        bridge = MLflowBridge(config=model_deployment_config)
        # Sync and deploy
        bridge.sync_and_deploy()
          


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        obj = ModelDeploymentPipeline()
        obj.main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\n{'='*80}")
    except Exception as e:
        logger.exception(e)
        import sys
        sys.exit(1) 