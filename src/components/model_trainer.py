'''
    *   model_trainer file typically contains the code that is responsible for training the machine learning model.
    *   model_trainer file might handle:
        -   Model Selection: Choosing the appropriate machine learning algorithm for the problem at hand1.
        -   Model Training: Feeding the training data to the model and adjusting the model’s parameters based on the training data1.
        -   Model Evaluation: Assessing the performance of the model on a separate validation dataset1.
        -   Hyperparameter Tuning: Fine-tuning the model’s hyperparameters to improve its performance1.
        -   Model Saving: Saving the trained model for later use in prediction1.

'''

import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting the trainig and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "Knn": KNeighborsRegressor(),
                "XGBR": XGBRegressor(),
                "catboost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor(),
            }

            model_report:dict=evaluate_model(X_train=X_train,
                                             y_train=y_train,
                                             X_test=X_test,
                                             y_test=y_test,
                                             models=models)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_mode = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info("Best model found")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_mode
            )

            predicted=best_mode.predict(X_test)
            r2score=r2_score(y_test,predicted)
            return r2score

        except Exception as e:
            raise CustomException(e,sys)        