import os
import sys

from src.exception import CustomException
from src.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
import pandas as pd
import numpy as np
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation=DataTransformationConfig()
    def get_transformation_object(self):
        '''This funcction is reponsible for data transformation'''
        try:
            categorical_features=[
                'Country',
                'Gender',
                'Demographics_Question',
                'Demographics_Response',
                'Question'
            ]

            
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical columns encoding completed")
            preprocessor=ColumnTransformer(
                [
                    ('cat_pipeline',cat_pipeline,categorical_features)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_set=pd.read_csv(train_path)
            test_set=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_transformation_object()

            target_column='Value'
            input_feature_train_df=train_set.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_set[target_column]
            input_feature_test_df=test_set.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_set[target_column]

            logging.info("Applying preprocessing object on training and testing dataframes")
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = input_feature_train_arr.toarray() 
                input_feature_test_arr = input_feature_test_arr.toarray()
            
            target_feature_train_df = np.array(target_feature_train_df).reshape(-1, 1) 
            target_feature_test_df = np.array(target_feature_test_df).reshape(-1, 1)

            train_arr = np.c_[input_feature_train_arr, target_feature_train_df] 
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df]

            logging.info("Saved Preprocessing object")
            save_object(file_path=self.data_transformation.preprocessor_obj_file,obj=preprocessing_obj)

            return (
                train_arr,
                test_arr,
                self.data_transformation.preprocessor_obj_file
            )

        except Exception as e:
            raise CustomException(e,sys)