import os
import logging
import pickle
import pandas as pd
import numpy as np
from model import CustomLinearModel
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split

class Trainer():
    """"""
    def __init__(self, df: pd.DataFrame, models_basedir: str, model_filename: str,  logger: logging.Logger, model: CustomLinearModel):
        self.df = df
        self.model = model
        self.__logger = logger
        self.__models_basedir = models_basedir
        self.__model_filename = model_filename
        self.__drop_columns = ["user", "device"]
        self.__target_defined = False
        self.__model_fitted = False
    
    def run(self):
        self.__define_target()
        self.__train_test_score()
        self.__fit()
        self.__save()

        return self.df
    
    def __define_target(self):
        self.target = self.df["revenue"]
        self.__target_defined = True
        self.df.drop("revenue", inplace=True, axis=1)

    def __fit(self):
        if self.__target_defined:
            self.model.fit(self.df.drop(self.__drop_columns, axis=1), self.target)
            self.__model_fitted = True
        else:
            raise ValueError("Target did not define")

    def __save(self):
        """
        Save model to disk
        """
        if self.__model_fitted:
            saved_path = os.path.join(self.__models_basedir, self.__model_filename)
            with open(os.path.join(self.__models_basedir, self.__model_filename), 'wb') as f:
                pickle.dump(self.model, f)
            self.__logger.info(f"Model saved as '{saved_path}'")
        else:
            raise ValueError("Model did not fit")

    def __cross_val_score(self):
        if self.__target_defined:
            metrics = cross_val_score(self.model, self.df.drop(self.__drop_columns, axis=1), self.target, scoring="neg_mean_absolute_error", cv=3)
            self.__logger.info(f"Cross-val score result - {metrics}")
        else:
            raise ValueError("Target did not define")

    def __train_test_score(self):
        if self.__target_defined:
            train_df, test_df, train_target, test_target = train_test_split(self.df, self.target, test_size=0.3, random_state=42)
            self.model.fit(train_df.drop(self.__drop_columns, axis=1), train_target)
            pred = self.model.predict(test_df.drop(self.__drop_columns, axis=1))
            self.__logger.info(f"Mean absolute erros on test - {mean_absolute_error(np.exp(test_target), np.exp(pred))}")
        else:
            raise ValueError("Target did not define")