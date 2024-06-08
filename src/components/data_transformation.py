'''
   Data Transformation is the process of converting raw data into a more suitable format or structure for analysis, to improve its quality and make it compatible with the requirements of a particular task or system.
   It’s an important step.
   It may in,cude steps such as:
   *    data cleaning 
   *    handling null values
   *    converting categorical to numerical
   
'''
import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

import numpy as np
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifact','preprocessor.pkl')
class DataTransformation:
   def __init__(self):
      self.data_tranformation_config=DataTransformationConfig()

   def get_data_transformer_object(self):
      #This function is responsible for data transformation
      try:
         numerical_cols = ["reading score","writing score"]
         categorical_cols = [
            "gender",
            "race/ethnicity",
            "parental level of education",
            "lunch",
            "test preparation course"
         ]
         num_pipeline=Pipeline(
            steps=[
               ("imputer",SimpleImputer(strategy="median")),
               ("scaler",StandardScaler())
            ]
         )
         cat_pipeline=Pipeline(
            steps=[
               ("imputer",SimpleImputer(strategy="most_frequent")),
               ("one_hot_encoder",OneHotEncoder()),
               ("scaler",StandardScaler(with_mean=False))
            ]
         )
         logging.info(f"categorical cols: {categorical_cols}")
         logging.info(f"numerical cols: {numerical_cols}")

         preprocessor=ColumnTransformer(
            [
               ("num_pipeline",num_pipeline,numerical_cols),
               ("cat_pipeline",cat_pipeline,categorical_cols)
            ]
         )

         return preprocessor
      except Exception as e:
         raise CustomException(e,sys)

   def initiate_data_transform(self,train_path,test_path):
      try:
         train_df=pd.read_csv(train_path)
         test_df=pd.read_csv(test_path)

         logging.info("Read train and test data completed")

         logging.info("Obtaining pre processing object")

         preprocess_obj=self.get_data_transformer_object()

         target_col_name = "math score"
         numerical_cols = ["reading score","writing score"]

         input_feature_train_df=train_df.drop(columns=[target_col_name],axis=1)
         target_feature_train_df=train_df[target_col_name]
         
         input_feature_test_df=test_df.drop(columns=[target_col_name],axis=1)
         target_feature_test_df=test_df[target_col_name]

         logging.info("Applying preprocessing obj on traing df and test df")

         input_feature_train_arr=preprocess_obj.fit_transform(input_feature_train_df)
         input_feature_test_arr=preprocess_obj.transform(input_feature_test_df)
         
         train_arr=np.c_[
            input_feature_train_arr,np.array(target_feature_train_df)
         ]         
         test_arr=np.c_[
            input_feature_test_arr,np.array(target_feature_test_df)
         ]

         logging.info("saved prepocessing object.")

         save_object(
            file_path=self.data_tranformation_config.preprocessor_obj_file_path,
            obj=preprocess_obj
         )

         return(
            train_arr,test_arr,self.data_tranformation_config.preprocessor_obj_file_path
         )

      except Exception as e:
         raise CustomException(e,sys)