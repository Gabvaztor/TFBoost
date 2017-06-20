"""
Normally, this files contains all necessary code to execute successfully the solution of the problem
but in this case (because this version is not stable) all code is in "TFModel_backup.py" file.
"""

# TODO Define Code
"""
TFBooster Code to solve problem
"""
setting_object = SettingsObject.Settings(Dictionary.string_settings_german_signal_path)

path_train_and_test_images = [setting_object.train_path,setting_object.test_path]
number_of_classes = 59 # Start in 0
percentages_sets = None  # Example
labels_set = [Dictionary.string_labels_type_option_hierarchy]
is_an_unique_csv = False  # If this variable is true, then only one CSV file will be passed and it will be treated like
# trainSet, validationSet(if necessary) and testSet
known_data_type = ''  # Contains the type of data if the data file contains an unique type of data. Examples: # Number
# or Chars.

reader_features = tfr.ReaderFeatures(set_data_files = path_train_and_test_images,number_of_classes = number_of_classes,
                                      labels_set = labels_set,
                                      is_unique_csv = is_an_unique_csv,known_data_type = known_data_type,
                                      percentages_sets = percentages_sets)

"""
Creating Reader from ReaderFeatures
"""
tf_reader = tfr.Reader(reader_features = reader_features)  # Reader Object with all information

"""
Getting train, validation (if necessary) and test set.
"""
test_set = tf_reader.test_set  # Test Set
train_set = tf_reader.train_set  # Train Set
del reader_features
del tf_reader

models = models.TFModels(input=train_set[0],test=test_set[0],
                         input_labels=train_set[1],test_labels=test_set[1],
                         number_of_classes=number_of_classes, setting_object=setting_object)
models.convolution_model_image()