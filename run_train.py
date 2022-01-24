import os
from posixpath import dirname
from pyexpat import model
import hydra
from omegaconf import DictConfig, OmegaConf
from data import DataLoader, DataPreparator
import pandas as pd
from model import CustomLinearModel

import logging

from train import Trainer

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config")
def run_pipeline(cfg : DictConfig) -> None:
    logger.info("Score pipeline started")

    logger.info("Loading data")
    Loader = DataLoader(os.path.join(dirname, cfg["data_basedir"]), cfg["train_data_filename"], logger)
    df = Loader.run()
    logger.info(f"Data loaded with shape {df.shape}")

    logger.info("Preparing data")
    Data_prep = DataPreparator(df, logger, os.path.join(dirname, cfg["encoder_basedir"]), mode="train")
    df = Data_prep.run()
    logger.info("Data prepared")

    model = CustomLinearModel()
    logger.info("Training model")
    Model_train = Trainer(df, os.path.join(dirname, cfg["models_basedir"]), cfg["model_filename"], logger, model)
    df = Model_train.run()
    logger.info("Model trained")

if __name__ == "__main__":
    dirname = os.path.dirname(os.path.realpath(__file__))
    run_pipeline()