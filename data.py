import os
import logging
from xml.dom import NotFoundErr
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle

class DataLoader():
    """ Data loader """
    def __init__(self, data_basedir: str, data_filename: str, logger: logging.Logger):
        self.__data_filepath = os.path.join(data_basedir, data_filename)
        self.__logger = logger
    
    def run(self):
        self.__logger.info(f"Load data from file - {self.__data_filepath}")
        df = pd.read_csv(self.__data_filepath)

        return df

class DataPreparator():
    """ Data preparation class for both train and score modes """
    def __init__(self, df: pd.DataFrame, logger: logging.Logger, encoders_basedir: str, mode: str="score"):
        self.df = df
        self.__logger = logger
        self.__encoders_basedir = encoders_basedir
        self.__columns_for_encoding = ["region"]
        self.__target_column = "revenue"
        self.__columns_for_log = ["trade_profit_usd", "trades_count"]
        self.__columns_for_drop = ["trade_sum_usd", "withdrawal_count"]
        self.__encoders = {col: None for col in self.__columns_for_encoding}
        self.mode = mode
        if self.mode not in ["train", "score"]:
            raise ValueError("Mode variable must be 'train' or 'score'")
    
    def run(self):
        self.__logger.info(f"Start data preparation. All columns - \n{self.df.columns}")
        if self.mode == "score":
            # self.__find_encoders()
            self.__encode_transform()
            self.__fillna()
            self.__log()
            self.__drop()
        else:
            self.__prepare_target()
            self.__encode_fit()
            self.__encode_transform()
            self.__fillna()
            self.__log()
            self.__drop()
        self.__logger.info(f"End data preparation. All columns - \n{self.df.columns}")

        return self.df

    def __prepare_target(self):
        self.__logger.info(f"Start target preparation. Target column - {self.__target_column}")
        self.df[self.__target_column] = np.log(self.df[self.__target_column] + 1)

    def __fillna(self):
        self.__logger.info(f"Start fill NaN values.")
        # 0 - median of 'trade_profit_usd'-column
        self.df["trade_profit_usd"] = self.df["trade_profit_usd"].fillna(0)
        # self.df["trade_profit_usd"] = self.df["trade_profit_usd"].fillna(self.df["trade_profit_usd"].meadian())
    
    def __log(self):
        self.__logger.info(f"Start log transformation. Columns for log transformation - {self.__columns_for_log}")
        for col in self.__columns_for_log:
            self.df[col] = np.log(self.df[col] + 1)

    def __find_encoders(self):
        for col in self.__columns_for_encoding:
            if ~os.path.isfile(os.path.join(self.__encoders_basedir, col + "_encoder")):
                raise NotFoundErr(f"""Not found encoder for column - {col}. 
                    File {os.path.join(self.__encoders_basedir, col + "_encoder")} does not exist""")

    def __encode_transform(self):
        self.__logger.info(f"""Start encode transformation. Columns for encoding - {self.__columns_for_encoding}.
        All encode files must be in folder '{self.__encoders_basedir}' and must be named 'column_name' + '_encoder'""")
        for col in self.__columns_for_encoding:
            with open(os.path.join(self.__encoders_basedir, col + "_encoder"), 'rb') as f:
                self.__encoders[col] = pickle.load(f)
            oh_features = self.__encoders[col].transform(self.df[col].values.reshape(-1, 1)).toarray()
            for i, category in enumerate(self.__encoders[col].categories_[0]):
                self.df[col + '_' + category] = oh_features[:, i]
            self.df.drop(col, axis=1, inplace=True)

    def __encode_fit(self):
        self.__logger.info(f"""Start encode fit. Columns for encoding - {self.__columns_for_encoding}.
        All encoders will be saved to folder '{self.__encoders_basedir}'""")
        for col in self.__columns_for_encoding:
            self.__encoders[col] = OneHotEncoder(handle_unknown='ignore')
            self.__encoders[col].fit(self.df[col].values.reshape(-1, 1))
        if ~os.path.exists(self.__encoders_basedir):
            os.mkdir(self.__encoders_basedir)
        with open(os.path.join(self.__encoders_basedir, col + "_encoder"), 'wb') as f:
            pickle.dump(self.__encoders[col], f)
        self.__logger.info(f"""All encoders saved""")

    def __drop(self):
        self.__logger.info(f"Dropping columns {self.__columns_for_drop}")
        for col in self.__columns_for_drop:
            self.df.drop(col, axis=1, inplace=True)