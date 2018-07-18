# -*- coding: utf-8 -*-
"""
This class contains useful functions
"""

"""
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
"""
"""LOCAL IMPORTS"""
from TFBoost.TFEncoder import *

'''Time library'''
import time

'''OS'''
import os

"""Numpy"""
import numpy as np

""" Shutil for copy and moves files"""
from shutil import *

""" TO SOLVE sum(list) not exact problem"""
from decimal import Decimal

'''
To install pandas: pip3 install pandas
'''
import pandas as pd

"""TensorFlow"""
import tensorflow as tf

"""Json"""
import json

import traceback
def pt(title=None, text=None):
    """
    Use the print function to print a title and an object coverted to string
    :param title:
    :param text:
    """
    if text is None:
        text = title
        title = Dictionary.string_separator
    else:
        title += ':'
    print(str(title) + " \n " + str(text))


def timed(method):
    """
    This method print a method name and the execution time.
    Normally will be used like decorator
    :param A method
    :return: The method called
    """
    def inner(*args, **kwargs):
        start = time.time()
        try:
            return method(*args, **kwargs)
        finally:
            methodStr = str(method)
            pt("Running time method:" + str(methodStr[9:-23]),
               str(time.strftime("%Hh%Mm%Ss", time.gmtime((time.time() - start)))))
    return inner


def clear_console():
    os.system('cls')


def number_neurons(total_input_size, input_sample_size, output_size, alpha=1):
    """
    :param total_input_size: x
    :param input_sample_size: x
    :param output_size: x
    :param alpha: x
    :return: number of neurons for layer
    """
    # TODO Finish docs
    return int(total_input_size / (alpha * (input_sample_size + output_size)))


def write_string_to_pathfile(string, filepath):
    """
    Write a string to a path file
    :param string: string to write
    :param path: path where write
    """
    try:
        create_directory_from_fullpath(filepath)
        file = open(filepath, 'w+')
        file.write(str(string))
    except:
        raise ValueError(Errors.write_string_to_file)


def write_json_to_pathfile(json, filepath):
    """
    Write a string to a path file
    :param string: string to write
    :param path: path where write
    """
    try:
        create_directory_from_fullpath(filepath)
        with open(filepath, 'w+') as file:
            # file = open(filepath, 'w+')
            file.write(str(json))
    except:
        raise ValueError(Errors.write_string_to_file)


def recurrent_method_pass_true_or_false(question, method):
    """

    :param question:
    :param method:
    :return:
    """
    # TODO (@gabvaztor) Docs
    response = False
    pt(Dictionary.string_get_response)
    save = str(input(question + " ")).upper()
    if save == Dictionary.string_char_Y:
        response = True
    elif save != Dictionary.string_char_N:
        method()
    return response


def recurrent_ask_to_save_model():
    """
    Wait user to get response to save a model
    :return: 
    """
    method = recurrent_ask_to_save_model
    response = recurrent_method_pass_true_or_false(question=Dictionary.string_want_to_save,
                                                   method=method)
    return response


def recurrent_ask_to_continue_without_load_model():
    """
    Wait user to get response to save a model
    :return: 
    """
    method = recurrent_ask_to_continue_without_load_model
    response = recurrent_method_pass_true_or_false(question=Dictionary.string_want_to_continue_without_load,
                                                   method=method)
    return response


def file_exists_in_path_or_create_path(filepath):
    """
    Check if filepath exists and, if not, it creates the dir
    :param filepath: the path to check
    :return True if exists filepath, False otherwise
    """
    try:
        create_directory_from_fullpath(filepath)
        if os.path.exists(filepath):
            return True
        else:
            return False
    except:
        raise ValueError(Errors.check_dir_exists_and_create)


def factorial(num):
    """
    Factorial of a number. Recursive.
    :param num: Number
    :return: Factorial
    """
    if num > 1:
        num = num * factorial(num - 1)
    return num


def create_directory_from_fullpath(fullpath):
    """
    Create directory from a fullpath if it not exists.
    """
    # TODO (@gabvaztor) Check errors
    directory = os.path.dirname(fullpath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def create_file_from_fullpath(fullpath):
    """
    Create file from a fullpath if it not exists.
    """
    # TODO (@gabvaztor) Check errors
    if not os.path.exists(fullpath):  # To create file
        file = open(fullpath, 'w+')
        file.close()


def create_historic_folder(filepath, type_file, test_accuracy=""):
    """
    Used when filepath exists to create a folder with actual_time to historicize
    :param filepath: file to save  
    :param type_file: Type of file (Information or Configuration)
    """

    # TODO (gabvaztor) Using new SettingObject path
    actual_time = str(time.strftime("%Y-%m-%d_%Hh%Mm%Ss", time.gmtime(time.time())))
    directory = os.path.dirname(filepath)
    filename = actual_time + "_" + os.path.basename(filepath)
    low_stripe = ""
    if test_accuracy and test_accuracy is not "":
        low_stripe = "_"
    information_folder = "\\History_Information\\" + type_file + "\\" + str(test_accuracy) + low_stripe + \
                         actual_time + "\\"
    folder = directory + information_folder
    create_directory_from_fullpath(folder)
    return folder + filename


def get_directory_from_filepath(filepath):
    return os.path.dirname(filepath)


def get_filename_from_filepath(filepath):
    return os.path.basename(filepath)


def preprocess_lists(lists, index_to_eliminate=2):
    """
    Preprocess each list reducing between x(=2 by defect) the number of samples 
    :param lists: Amount of lists to be reduced
    :param index_to_eliminate: represent which index module must be deleted to the graph. For example, if 
    list len is 60 and 'index_to_eliminate' is 2, then all element with a pair index will be eliminated, remaining 30.
    :return: Process_lists
    """
    processed_lists = []
    for list_to_process in lists:
        if list_to_process and len(list_to_process) > index_to_eliminate:
            process_list = [i for i in list_to_process if list_to_process.index(i) % index_to_eliminate == 0]
            processed_lists.append(process_list)
        else:
            processed_lists.append(list_to_process)
    return processed_lists


def save_numpy_arrays_generic(folder_to_save, numpy_files, names=None):
    """
    Save the accuracies and losses into a type_file folder.
    names and numpy_files are two list.
    :param folder_to_save:
    :param numpy_files:
    :param names: Must have same size than numpy_files

    """
    # TODO (@gabvaztor) finish Docs
    create_directory_from_fullpath(folder_to_save)
    for index in range(len(numpy_files)):
        if names:
            np.save(folder_to_save + names[index], numpy_files[index])
        else:
            name_file = Dictionary.filename_numpy_default
            np.save(folder_to_save + name_file + str(index + 1), numpy_files[index])


def load_numpy_arrays_generic(path_to_load, names):
    """
    :param path_to_load: 
    :param names: 
    :return: 
    """
    # TODO (@gabvaztor) DOCS
    files_to_return = []
    npy_extension = Dictionary.string_npy_extension
    for i in range(len(names)):
        if file_exists_in_path_or_create_path(path_to_load + names[i] + npy_extension):
            file = np.load(path_to_load + names[i] + npy_extension)
            files_to_return.append(file)
        else:
            raise Exception("File does not exist")
    return files_to_return


def load_accuracies_and_losses(path_to_load, flag_restore_model=False):
    """
    
    :param path_to_load: path to load the numpy accuracies and losses
    :return: accuracies_train, accuracies_validation, loss_train, loss_validation
    """
    # TODO (@gabvaztor) Docs
    accuracies_train, accuracies_validation, loss_train, loss_validation = [], [], [], []
    if flag_restore_model:
        try:
            npy_extension = Dictionary.string_npy_extension
            filename_train_accuracies = Dictionary.filename_train_accuracies + npy_extension
            filename_validation_accuracies = Dictionary.filename_validation_accuracies + npy_extension
            filename_train_losses = Dictionary.filename_train_losses + npy_extension
            filename_validation_losses = Dictionary.filename_validation_losses + npy_extension
            if file_exists_in_path_or_create_path(path_to_load + filename_train_accuracies):
                accuracies_train = list(np.load(path_to_load + filename_train_accuracies))
                accuracies_validation = list(np.load(path_to_load + filename_validation_accuracies))
                loss_train = list(np.load(path_to_load + filename_train_losses))
                loss_validation = list(np.load(path_to_load + filename_validation_losses))
        except Exception:
            pt("Could not load accuracies and losses")
            accuracies_train, accuracies_validation, loss_train, loss_validation = [], [], [], []

    return accuracies_train, accuracies_validation, loss_train, loss_validation


def load_4_numpy_files(path_to_load, names_to_load_4):
    # TODO (@gabvaztor) Docs
    npy_extension = Dictionary.string_npy_extension
    for i in range(0, 4):
        file_exists_in_path_or_create_path(path_to_load + names_to_load_4[i])
    file_1 = np.load(path_to_load + names_to_load_4[0] + npy_extension)
    file_2 = np.load(path_to_load + names_to_load_4[1] + npy_extension)
    file_3 = np.load(path_to_load + names_to_load_4[2] + npy_extension)
    file_4 = np.load(path_to_load + names_to_load_4[3] + npy_extension)

    return file_1, file_2, file_3, file_4


def save_and_restart(path_to_backup):
    """
    Save and restart all progress. Create a "Zip" file from "Models" folder and, after that, remove it. 
    :param path_to_backup: Path to do a backup and save it in a different folder
    """
    actual_time = str(time.strftime("%Y-%m-%d_%Hh%Mm%Ss", time.gmtime(time.time())))
    to_copy = get_directory_from_filepath(path_to_backup) + "\\"
    to_paste = get_directory_from_filepath(
        get_directory_from_filepath(to_copy)) + "\\" + "Models_Backup(" + actual_time + ")"
    pt("Doing Models backup ...")
    # Do backup
    make_archive(to_paste, 'zip', to_copy)
    pt("Backup done successfully")
    # Do remove
    pt("Removing Models folder...")
    rmtree(to_copy)
    pt("Models removed successfully")


def convert_to_decimal(percentages_sets):
    """
    Convert all values from a list to Decimal and return the sum
    To fix "sum(list)" not exact return problem
    :param percentages_sets: list of percent (decimal values < 1)
    :return: list sum
    """
    values = []
    for value in percentages_sets:
        decimal = Decimal(str(value))
        values.append(decimal)
    total_sum = sum(values)
    return total_sum


@timed
def save_submission_to_csv(path_to_save, dictionary):
    print('Saving submission...')
    pt("Getting keys...")
    keys = list(dictionary.keys())
    pt("Getting values...")
    values = list(dictionary.values())
    submission = pd.DataFrame({'Pages_Date': keys, 'Visits': values})
    pt("Saving to csv in path...")
    submission.to_csv(path_to_save, index=False, encoding='utf-8')
    pt("Save successfully")


def smape(y_true, y_prediction):
    tot = tf.reduce_sum(y_true)
    map = tf.truediv((tf.subtract(tf.abs(y_prediction), tf.abs(y_true))),
                     (tf.add(tf.abs(y_true), tf.abs(y_prediction))) / 2.0)
    smape = tf.realdiv(100 * map, tot)
    return smape


def mean_difference_percentages(y_true, y_prediction):
    # Total from real
    total_real = tf.reduce_sum(y_true)
    # For each element, we calculate the abs difference and, after that, we calculate the percent respect total.
    difference = tf.abs(tf.subtract(y_true, y_prediction))
    mean_difference = tf.truediv((difference * 100), total_real)
    return mean_difference


def rmse(y_true, y_prediction):
    return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, y_prediction))))


def mape(y_true, y_prediction):
    tot = tf.reduce_sum(y_true)
    wmape = tf.realdiv(tf.reduce_sum(tf.abs(tf.subtract(y_true, y_prediction))), tot) * 100  # /tot
    return (wmape)


def mae(y_true, y_prediction, batch_size):
    total_sum_y_true = tf.reduce_sum(y_true)
    total_sum_y_prediction = tf.reduce_sum(y_prediction)
    error = tf.sqrt(tf.abs((tf.subtract(y_true, y_prediction))))
    return error


def convert_to_numpy_array(to_convert_to_numpy_array_list):
    to_return = []
    if to_convert_to_numpy_array_list:
        for element in to_convert_to_numpy_array_list:
            element = np.asarray(element)
            to_return.append(element)
    return to_return


def is_json(my_json):
    try:
        json_object = json.loads(str(my_json))
    except ValueError:
        return False
    return True

def class_properties(object, attributes_to_delete=None):
    """
    Return a string with actual object features without not necessaries
    :param attributes_to_delete: represent witch attributes set must be deleted.
    :return: A copy of class.__dic__ without deleted attributes
    """
    dict_copy = object.__dict__.copy()  # Need to be a copy to not get original class' attributes.
    # Remove all not necessaries values
    if attributes_to_delete:
        for x in attributes_to_delete:
            del dict_copy[x]
    return dict_copy

def object_to_json(object, attributes_to_delete=None):
    """
    Convert class to json with properties method.
    :param attributes_to_delete: String set with all attributes' names to delete from properties method
    :return: sort json from class properties.
    """
    try:
        object_dictionary = class_properties(object=object, attributes_to_delete=attributes_to_delete)
        json_string = json.dumps(object, default=lambda m: object_dictionary, sort_keys=True, indent=4)
    except Exception as e:
        pt(e)
        pt(traceback.print_exc())
        raise ValueError("PARAR")
    return json_string

def show_actual_path():
    pt("Actual Path", os.path.dirname(os.path.abspath(__file__)))
    pt("Actual Path", os.getcwd())

def get_temp_file_from_fullpath(fullpath):
    """
    Create and return the temp fullpath from another fullpath

    Args:
        fullpath: the fullpath

    Returns: the created temp fullpath
    """
    basename = os.path.basename(fullpath)
    path = os.path.dirname(fullpath) + "\\temp\\"
    create_directory_from_fullpath(path)

    return path + basename

def read_changing_data(open_file):
    #open_file.seek(0, 2)
    while True:
        line = open_file.readline().strip()
        if not line:
            time.sleep(0.1)
            continue
        open_file.flush()
        yield line

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')

    # Print New Line on Complete
    if iteration == total:
        print()

def get_files_from_path(paths, ends_in=None):
    """
    Get all files in paths. If ends_in, return only those files ends in a string
    Args:
        paths: Paths to search (could be a list of paths)
        ends_in: ends in. For example, a extension (".png", ".dat")

    Returns: all file_names
    """
    if type(paths) == type(""):
        paths = [paths]
    for path in paths:
        if not os.path.exists(path):
            raise ("Path does not exist.")
        else:
            for root, dirs, files in os.walk(path):
                for count_number, file_name in enumerate(files):
                    full_path = os.path.join(root, file_name)
                    name = os.path.splitext(file_name)[0]
                    if ends_in:
                        if file_name.endswith(ends_in):
                            yield full_path, root, name
                    else:
                        yield full_path, root, name