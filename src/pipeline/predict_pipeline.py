import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,days):
        try:
            model_path='artifacts\model.pkl'
            model=load_object(file_path=model_path)
            preds=model.forecast(days)
            predictions = [i for i in preds]
            return predictions
        
        except Exception as e:
            raise CustomException(e,sys)
        


class CustomData:
    def __init__( self,forecast_days: int):
        self.forecast_days = forecast_days

    def get_data_web(self):
        try:
            return int(self.forecast_days)
        except Exception as e:
            raise CustomException(e, sys)