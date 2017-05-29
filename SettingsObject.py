"""
This class is used in kaggle competitions
"""

import json
import os
from TFBoost.TFEncoder import Dictionary
from UsefulTools.UtilsFunctions import *

class Settings():
    _train_path = "D:\\"
    _test_path = "D:\\"
    _model_path = "D:\\"
    _submission_path = "D:\\"
    _configuration_path = "D:\\"
    _information_path= "D:\\"
    _path = "SETTINGS.json"
    string_train_path = "TRAIN_DATA_PATH"
    string_test_path = "TEST_DATA_PATH"
    string_model_path = "MODEL_PATH"
    string_submission_path = "SUBMISSION_PATH"
    string_configuration_path = "CONFIGURATION_PATH"
    string_information_path = "INFORMATION_PATH"

    @property
    def train_path(self): return self._train_path

    @property
    def test_path(self): return self._test_path

    @property
    def model_path(self): return self._model_path

    @property
    def submission_path(self): return self._submission_path

    @property
    def configuration_path(self): return self._configuration_path

    @property
    def information_path(self): return self._information_path

    @property
    def path(self): return self._path

    @train_path.setter
    def train_path(self, value): self._train_path=value

    @test_path.setter
    def test_path(self, value): self._test_path=value

    @model_path.setter
    def model_path(self, value): self._model_path=value

    @submission_path.setter
    def submission_path(self, value): self._submission_path=value

    @configuration_path.setter
    def configuration_path(self, value): self._configuration_path=value

    @information_path.setter
    def information_path(self, value): self._information_path=value

    @path.setter
    def path(self, value): self._path = value

    def __init__(self, path):
        self._path = path
        self._load_settings()
    def __str__(self):
        to_print = "\n".join([self.string_train_path,
                              self.train_path,
                              self.string_test_path,
                              self.test_path,
                              self.string_model_path,
                              self.model_path,
                              self.string_submission_path,
                              self.submission_path])
        return to_print
    def _load_settings(self):
        # Change with your setting file if necessary
        with open(self.path) as json_data:
            settings = json.load(json_data)
            self.train_path=settings[self.string_train_path]
            self.test_path=settings[self.string_test_path]
            self.model_path=settings[self.string_model_path]
            self.submission_path=settings[self.string_submission_path]
            self.configuration_path=settings[self.string_configuration_path]
            self.information_path=settings[self.string_information_path]

    def load_actual_information(self):
        """

        :return:
        """
        # TODO (@gabvaztor) DOCs
        create_directory_from_fullpath(self.information_path)
        create_file_from_fullpath(self.information_path)
        if os.stat(self.information_path).st_size != 0:
            with open(self.information_path) as json_configuration:
                dict = json.load(json_configuration)
                configuration = Configuration(dict)
                return configuration
        else:
            configuration = None
            return configuration

class Configuration():
     def __init__(self, json_content):
         for key, value in json_content.items():
             self.__dict__[key] = value
