"""
This class is used in kaggle competitions
"""

import json

from pathlib import Path
import os

from TFBoost.TFEncoder import Dictionary
class Settings():
    _train_path = "D:\\"
    _test_path = "D:\\"
    _model_path = "D:\\"
    _submission_path = "D:\\"
    _information_path= "D:\\"
    _path = "SETTINGS.json"
    string_train_path = "TRAIN_DATA_PATH"
    string_test_path = "TEST_DATA_PATH"
    string_model_path = "MODEL_PATH"
    string_submission_path = "SUBMISSION_PATH"
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

    @information_path.setter
    def information_path(self, value): self._information_path=value

    @path.setter
    def path(self, value): self._path = value

    def __init__(self, path):
        if not path:
            path = Dictionary.string_settings_path
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
            self.information_path=settings[self.string_information_path]

    def load_actual_configuration(self):
        """

        :return:
        """
        # TODO DOCs
        directory = os.path.dirname(self.information_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        if not os.path.exists(self.information_path):
            open(self.information_path, 'w+')
        if os.stat(self.information_path).st_size != 0:
            with open(self.information_path) as json_configuration:
                configuration = json.load(json_configuration)
                configuration.num_trains_count = configuration['_num_trains_count']
                configuration.train_dropout = configuration['_train_dropout']
                configuration.epoch_numbers = configuration['_epoch_numbers']
                configuration.third_label_neurons = configuration['_third_label_neurons']
                configuration.shuffle_data = configuration['_shuffle_data']
                configuration.input_rows_numbers = configuration['_input_rows_numbers']
                configuration.second_label_neurons = configuration['_second_label_neurons']
                configuration.train_accuracy = configuration['_train_accuracy']
                configuration.test_size = configuration['_test_size']
                configuration.number_of_classes = configuration['_number_of_classes']
                configuration.input_size = configuration['_input_size']
                configuration.input_columns_numbers = configuration['_input_columns_numbers']
                configuration.kernel_size = configuration['_kernel_size']
                configuration.restore_model = configuration['_restore_model']
                configuration.learning_rate = configuration['_learning_rate']
                configuration.trains = configuration['_trains']
                configuration.batch_size = configuration['_batch_size']
                configuration.first_label_neurons = configuration['_first_label_neurons']
                configuration.test_accuracy = configuration['_test_accuracy']
                return configuration
        else:
            configuration = None
            return configuration