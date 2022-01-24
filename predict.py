import os
import logging
import pickle
import pandas as pd
import numpy as np

class Predictor():
    """"""
    def __init__(self, df: pd.DataFrame, models_basedir: str, model_filename: str,  logger: logging.Logger):
        self.df = df
        self.model = None
        self.__models_basedir = models_basedir
        self.__model_filename = model_filename
        self.__loaded = False
        self.__load_model()
        self.__prediction_columnname = "prediction"
        self.__drop_columns = ["user", "device"]
    
    def run(self):
        if self.__loaded:
            self.__predict()
        else:
            raise ValueError("Model didn't load")

        return self.df

    def __predict(self):
        self.df[self.__prediction_columnname] = np.exp(self.model.predict(self.df.drop(self.__drop_columns, axis=1)))
        return self.df

    def __load_model(self):
        """
        Load model from disk
        """
        with open(os.path.join(self.__models_basedir, self.__model_filename), 'rb') as f:
            self.model = pickle.load(f)
        self.__loaded = True