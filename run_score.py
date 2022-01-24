import os
import hydra
from omegaconf import DictConfig
from data import DataLoader, DataPreparator
import pandas as pd

import logging

from predict import Predictor


# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config")
def run_pipeline(cfg : DictConfig) -> None:
    logger.info("Score pipeline started")

    logger.info("Loading data")
    Loader = DataLoader(os.path.join(dirname, cfg["data_basedir"]), cfg["test_data_filename"], logger)
    df = Loader.run()
    logger.info(f"Data loaded with shape {df.shape}")

    logger.info("Preparing data")
    Data_prep = DataPreparator(df, logger, os.path.join(dirname, cfg["encoder_basedir"]))
    df = Data_prep.run()
    logger.info("Data prepared")

    logger.info("Scoring data")
    Model_predict = Predictor(df, os.path.join(dirname, cfg["models_basedir"]), cfg["model_filename"], logger)
    df = Model_predict.run()
    logger.info("Data scored")

    df[["user", "prediction"]].to_csv(os.path.join(dirname, cfg["scored_filename"]), index=False)

if __name__ == "__main__":
    dirname = os.path.dirname(os.path.realpath(__file__))
    run_pipeline()