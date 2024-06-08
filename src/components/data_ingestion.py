'''
    Data ingestion is the process of collecting and importing data from various sources into a machine-learning model. 
    It is the first step in building a machine-learning model and is crucial for the success of the model.
    Without proper data ingestion, the model will not be able to learn and make accurate predictions.

    The data_ingestion file could include code for:
    *   Connecting to data sources like databases or APIs.
    *   Reading data from files or streaming platforms.
    *   Cleaning and preprocessing the data.
    *   Splitting the data into training and testing sets.    
    
'''

import os
import sys 
from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
@dataclass
class DataIngestionConfig:
    #these are the inputs given to data ingestion component and it saves it in particular path
    train_data_path: str=os.path.join('artifact',"train.csv")
    test_data_path: str=os.path.join('artifact',"test.csv")
    raw_data_path: str=os.path.join('artifact',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the Data ingestion method or component")
        try:
            df = pd.read_csv('notebook\data\stud.csv') #getting the dataframe inside df variable
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) #storing my csv file as raw data in its path

            logging.info('Train test split initiated')
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42) #splitting the my dataframe df into train and test set

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) #storing the train and test set in the path
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Injestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()       
    train_data,test_data=obj.initiate_data_ingestion()

    data_tranformation=DataTransformation()
    data_tranformation.initiate_data_transform(train_data,test_data)