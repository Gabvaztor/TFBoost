"""
This class is used in kaggle competitions
"""

import json

class Settings():
    train_path = None
    test_path = None
    model_path = None
    submission_path = None
    string_train_path = "TRAIN_DATA_PATH"
    string_test_path = "TEST_DATA_PATH"
    string_model_path = "MODEL_PATH"
    string_submission_path = "SUBMISSION_PATH"

    def __init__(self):
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
        with open('SETTINGS.json') as json_data:
            settings = json.load(json_data)
            self.train_path=settings[self.string_train_path]
            self.test_path=settings[self.string_test_path]
            self.model_path=settings[self.string_model_path]
            self.submission_path=settings[self.string_submission_path]


s = Settings()

