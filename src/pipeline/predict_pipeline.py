import sys
import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path="artifacts\model.pkl"
            preprocessor_path='artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            pred=model.predict(data_scaled)
            return pred
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 Country:str,
                 Gender:str,
                 Demographics_Question:str,
                 Demographics_Response:str,
                 Question:str):
        self.Country=Country
        self.Gender=Gender
        self.Demographics_Question=Demographics_Question
        self.Demographics_Response=Demographics_Response
        self.Question=Question
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={
                'Country':[self.Country],
                'Gender':[self.Gender],
                'Demographics_Question':[self.Demographics_Question],
                'Demographics_Response':[self.Demographics_Response],
                'Question':[self.Question],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)
