#data ingestion is the file for reading the data from datbases and etc
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
#dataclass for auto generate methods like __init__ etc primarly used to store data
@dataclass
class DataIngestionConfig:
    train_data_path : str=os.path.join('artifacts',"train.csv")#data class helps where to save the train,test,raw data 
    test_data_path : str=os.path.join('artifacts',"test.csv")#artifact is folder 
    raw_data_path : str=os.path.join('artifacts',"raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv("notebook/data/StudentsPerformance.csv")

            logging.info("read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)# raw data to csv file

            logging.info("train test split initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state = 42)

            train_set.to_csv(self.ingestion_config.train_data_path,index= False,header =True)#saving the train file
            test_set.to_csv(self.ingestion_config.test_data_path,index= False,header =True)#saving the test file

            logging.info("ingestion of data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            raise CustomException(e,sys)


#initiate and  run 
if __name__== "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
