import os
import sys
# src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
# sys.path.append(src_path)

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, src_path)

from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
# used to create class variables

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer, ModelTrainerConfig



# in data ingestion, 
# inputs req:save - train data, test, raw data
# these inputs will be created in DataIngestion class
# any input required

@dataclass
class DataIngestionConfig:
    # this is the input 
    # later data ingestion will save the train/test/raw.csv to this path
    train_data_path:str= os.path.join("artifacts", "train.csv")
    test_data_path:str= os.path.join("artifacts", "test.csv")
    raw_data_path:str= os.path.join("artifacts", "raw.csv")
    # these r inputs we r giving to our dataingestion component
    # n now dataingestion component knows where to save train/test/raw data path


class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig() 
        
        # read data from databases
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        try:
            # read data from databases/api's/csv
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')
            # create folders for train/test/raw n combine dir path names
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Train test split initiated')
            # train_set, test_set=train_test_split(df, test_set=0.2, random_state=42)
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)



if __name__== "__main__":
    obj = DataIngestion()
    # obj.initiate_data_ingestion()
    train_data, test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, _= data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))



    










