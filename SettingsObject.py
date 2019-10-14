"""
This class is used in kaggle competitions
"""

import json
from UsefulTools.UtilsFunctions import *

class Settings():
    _train_path = "D:\\"
    _test_path = "D:\\"
    _model_path = "D:\\"
    _submission_path = "D:\\"
    _configuration_path = "D:\\"
    _information_path= "D:\\"
    _accuracies_losses_path= ""
    _history_information_path = ""
    _history_configuration_path = ""
    _labels_path = ""
    _path = "SETTINGS.json"
    _saved_dataset_path = ""
    string_train_path = "TRAIN_DATA_PATH"
    string_test_path = "TEST_DATA_PATH"
    string_model_path = "MODEL_PATH"
    string_submission_path = "SUBMISSION_PATH"
    string_configuration_path = "CONFIGURATION_PATH"
    string_information_path = "INFORMATION_PATH"
    string_accuracies_losses_path = "ACCURACIES_LOSSES_PATH"
    string_history_information_path = "HISTORY_INFORMATION_PATH"
    string_history_configuration_path = "HISTORY_CONFIGURATION_PATH"
    string_labels_path = "LABELS_PATH"
    string_saved_dataset_path = "SAVED_DATASET_PATH"
    # TODO (@gabvaztor) Create new variable that save "generic path". This means that you only have to specify only one
    # path and system will get (from a generic structure) other paths.

    # TODO (@gabvaztor) Configure with new paths
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

    @property
    def accuracies_losses_path(self): return self._accuracies_losses_path

    @accuracies_losses_path.setter
    def accuracies_losses_path(self, value): self._accuracies_losses_path=value

    @property
    def history_information_path(self): return self._history_information_path

    @history_information_path.setter
    def history_information_path(self, value): self._history_information_path=value

    @property
    def history_configuration_path(self): return self._history_configuration_path

    @history_configuration_path.setter
    def history_configuration_path(self, value): self._history_configuration_path=value

    @property
    def labels_path(self): return self._labels_path

    @labels_path.setter
    def labels_path(self, value): self._labels_path=value

    @property
    def saved_dataset_path(self): return self._saved_dataset_path

    @saved_dataset_path.setter
    def saved_dataset_path(self, value): self._saved_dataset_path=value

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
                              self.labels_path,
                              self.saved_dataset_path,
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
            self.accuracies_losses_path=settings[self.string_accuracies_losses_path]
            self.history_information_path=settings[self.string_history_information_path]
            self.history_configuration_path=settings[self.string_history_configuration_path]
            self.labels_path=settings[self.string_labels_path]
            self.saved_dataset_path=settings[self.string_saved_dataset_path]

    def load_actual_information(self):
        """
        Information path contains best accuracy to compare before save.
        """
        # TODO (@gabvaztor) DOCs
        configuration = None
        create_directory_from_fullpath(self.information_path)
        create_file_from_fullpath(self.information_path)
        if os.stat(self.information_path).st_size != 0:
            with open(self.information_path) as json_configuration:
                dict = json.load(json_configuration)
                configuration = Configuration(dict)
        return configuration

    def load_actual_configuration(self):
        """
        :return: configuration
        """
        # TODO (@gabvaztor) DOCS
        configuration = None
        create_directory_from_fullpath(self.configuration_path)
        create_file_from_fullpath(self.configuration_path)
        try:
            if os.stat(self.configuration_path).st_size != 0:
                with open(self.configuration_path) as json_configuration:
                    dict = json.load(json_configuration)
                    configuration = Configuration(dict)
        except Exception:
            input("Configuration problem: There is not a Configuration json file or this has nothing. Press Ok to"
                  "continue the execution and the file will be created automatically or stop the program and create for"
                  "your own")
        return configuration

class Configuration():
     def __init__(self, json_content):
         for key, value in json_content.items():
             self.__dict__[key] = value