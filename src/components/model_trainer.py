import os 
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer=ModelTrainerConfig()
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models= {
                "Linear Regression": LinearRegression(),
                "Lasso Regression": Lasso(),
                "Ridge Regression": Ridge(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "K Nearest Neighbors": KNeighborsRegressor(),
                "Xgboost": XGBRegressor(),
                "AdaBoost": AdaBoostRegressor()
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "Xgboost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost":{},
                "Lasso Regression": {},
                "Ridge Regression": {},
                "K Nearest Neighbors":{}
                
            }
            report:dict=evaluate_model(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,models=models,param=params)
            best_model_score=max(sorted(report.values()))
            best_model_name=list(report.keys())[list(report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("Best Model Found")

            save_object(file_path=self.model_trainer.trained_model_file_path,obj=best_model)

            predicted=best_model.predict(X_test)
            R2=r2_score(y_pred=predicted,y_true=y_test)
            return R2
        except Exception as e:
            raise CustomException(e,sys)