import os
import sys
from dataclasses import dataclass

import statsmodels
import statsmodels.api as sm
from scipy import stats
import numpy as np
from sklearn.metrics import mean_squared_error

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            y_train,y_test=(
                train_array[:],
                test_array[:]
            )

            model = sm.tsa.statespace.SARIMAX(y_train,order=(3,0,0))

            fitted_model,model_report=evaluate_models(y_train=y_train,y_test=y_test,model=model)

            logging.info(f"Model trained and evaluated")



            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=fitted_model
            )

            predicted=fitted_model.forecast(10)

            abs_rmse = mean_squared_error(y_test, predicted,squared = False)
            rel_rmse = abs_rmse/(np.max(y_train)-np.min(y_train))

            return rel_rmse
        

        except Exception as e:
            raise CustomException(e,sys)