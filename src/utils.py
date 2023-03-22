import os 
import sys
import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from sklearn.metrics import mean_squared_error
import numpy as np


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(y_train,y_test,model):
    try:
        report = {}
        # print(y_test)

        fitted_model = model.fit(disp=False)

        predicted = fitted_model.forecast(10)
        

        report['abs_rmse'] = np.sqrt(np.sum((predicted-y_test.reshape(-1))**2)/len(predicted))
        report['rel_rmse'] = report['abs_rmse']/(np.max(y_train)-np.min(y_train))

    except Exception as e:
        raise CustomException(e, sys)


    return fitted_model,report


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

