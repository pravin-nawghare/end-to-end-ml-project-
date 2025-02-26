# Model trainer + model pusher
import os
import sys

from dataclasses import dataclass
from datetime import datetime
from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,
                              GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Spliting training and test input data")

            x_train, y_train, x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Linear Regresor": LinearRegression(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "KNeighbors Regressor": KNeighborsRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "XGB Regressor": XGBRegressor(),
                "Decision Tree": DecisionTreeRegressor()
            }
            
            model_report:dict = evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,
                                               y_test=y_test, models= models)
            
            ##To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ##To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]       

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")

            logging.info("Best found model on both training and test set")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
                )

            predicted = best_model.predict(x_test)
            r2_scores = r2_score(y_test, predicted)
            
            return r2_scores
            
            
        except Exception as e:
            raise CustomException(e,sys)



