import os
import pandas as pd
from src.wine_quality_project import logger
from sklearn.model_selection import train_test_split
from src.wine_quality_project.config.configuration import DataTransfromationConfig


class DataTransformation:
    def __init__(self, config: DataTransfromationConfig):
        self.config=config

    def train_test_split(self):
        data = pd.read_csv(self.config.data_path)
        train, test = train_test_split(data, test_size=0.25)
        train.to_csv(os.path.join(self.config.root_dir, 'train.csv'), index=False)
        test.to_csv(os.path.join(self.config.root_dir, 'test.csv'), index=False)
        logger.info('splitting data into train and test sets')
        logger.info(train.shape)
        logger.info(test.shape)
        print(train.shape)
        print(test.shape)