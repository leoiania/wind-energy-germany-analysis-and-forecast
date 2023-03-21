import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        Function responsible for data transformation (i.e. scale centering around the mean and handling missing values)
        '''
        try:
            num_column = ["wind_generation_actual"]

            num_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")), # handling missing values
                ("scaler",StandardScaler(with_std=False)) # centers around mean
                ]
            )


            logging.info("Column centering completed")


            preprocessor = ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,num_column)
                ]
            )

            return preprocessor 
        
        except Exception as e:
            raise CustomException(e , sys)
            

    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")


            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()


            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            z = pd.concat([train_df,test_df],ignore_index=True)
            z_scal = preprocessing_obj.fit_transform(z)
            

            z_train = z_scal[:-10]
            z_test = z_scal[-10:]

            logging.info(f"Save preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (z_train,z_test,self.data_transformation_config.preprocessor_obj_file_path)
        except Exception as e:
            raise CustomException(e,sys)
        

