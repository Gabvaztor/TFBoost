import time

from Projects.SIL.src.DTO.ElectricClamp import ElectricClamp
from Projects.SIL.src.DTO.Humidity import Humidity
from Projects.SIL.src.DTO.Luminosity import Luminosity
from Projects.SIL.src.DTO.Presence import Presence
from Projects.SIL.src.DTO.Temperature import Temperature
from UsefulTools.UtilsFunctions import pt, printProgressBar, get_files_from_path, create_directory_from_fullpath, \
    file_exists_in_path_or_create_path, is_none, check_file_exists_and_change_name, filename_and_extension_from_fullpath
from AsynchronousThreading import execute_asynchronous_thread
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import matplotlib.dates as mdates
from datetime import datetime, date as typedate
import plotly.plotly as plotly
import numpy as np
import os
import gc
import multiprocessing
from Projects.SIL.src.Services.DataObject import DataObject, DataTypes, InfoDataObject, Measures, Sensor
from sys import getsizeof
import traceback



style_list = ['default', 'classic', 'Solarize_Light2', '_classic_test', 'bmh', 'dark_background', 'fast',
              'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright', 'seaborn-colorblind',
              'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted',
              'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks',
              'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblind10']
style_ = style_list[4]
pt(style_)
style.use(style=style_)

# Paths
clamp_filepath = "\\\\192.168.1.62\\miranda\\10_20180702.dat"
# clamp_filepath="E:\\SmartIotLabs\\DATA\\nEW\\10_20180702.dat"
global_sensor_filepath = "\\\\192.168.1.62\\miranda\\1_20180702.dat"
miranda_path = "\\\\192.168.1.220\\miranda\\"
miranda_path = "..\\..\\data\\temp\\"
#miranda_path = "F:\\Data_Science\\Projects\\Smartiotlabs\\Data\\"
# global_sensor_filepath="E:\\SmartIotLabs\\DATA\\nEW\\1_20180702.dat"
# Save image path
save_path = "E:\\SmartIotLabs\\AI_Department\\Data\\Sensors\\"
checkpoint_path = "E:\\SmartIotLabs\\AI_Department\\Data\\Sensors\\Checkpoints\\"
checkpoint_path = "..\\..\\data\\intermediate\\"

#figures = []
#graphs = []


delta_data = []
# Information
sensor_data_ids = []  # Contains the information of sensors and data_ids. The format is: ["sensorID_dataID", ...]
# Will be used to check when a type data was used or not.
data_objects = {}
saved_data_objects_doid = []
loaded_data_objects_doid = []
not_to_save_paths = []  # Contains all paths whose data_objects will not be saved

def add_to_graph(title=None, ylabel=None, ax=None):
    plt.title(title)
    plt.suptitle(title)
    plt.grid()

def calculate_deltas_from_data():
    for i, (sensor_data_id, data_object) in enumerate(data_objects.items()):
        if data_object.y:
            delta = max(data_object.y) - min(data_object.y)
            data_object.deltas.append(delta)
        elif data_object.multiple_y:
            for data_type in data_object.multiple_y:
                delta = max(data_type) - min(data_type)
                data_object.deltas.append(delta)
        else:
            delta_data.append(0)
            raise Exception("Data object has not values in 'y'")

def get_all_sensor_types(actual_sensor_types, path):
    """
    Args:
        actual_sensor_type: actual sensor types to be updated.

    Returns: sensor types with actual_sensor_types more all sensor types information
    """
    for fullpath, root, name in get_files_from_path(paths=path, ends_in=".dat"):
        sensor_id = int(name.split("_")[0])
        if sensor_id not in actual_sensor_types:
            actual_sensor_types.append(sensor_id)
    return actual_sensor_types

def update_data(load_data):
    sensor_types = []
    DataObjectReader(path=miranda_path, sensor_type=20, load_data=load_data).read_refresh_data()
    update_all_data = False
    if update_all_data:
        sensor_types = get_all_sensor_types(actual_sensor_types=sensor_types, path=miranda_path)
    #sensor_types.remove(20)
    for sensor_type in sensor_types:
        DataObjectReader(path=miranda_path, sensor_type=sensor_type, load_data=load_data).read_refresh_data()
    update_data_objects()
    return None

def sort_paths_by_date_in_basename(sorted_paths, sorted_with_dates, fullpath, name):
    date = datetime.strptime(name.split("_")[1], "%Y%m%d")
    if not sorted_paths:
        sorted_with_dates.append(date)
        sorted_paths.append(fullpath)
    else:
        sorted_with_dates.append(date)
        sorted_with_dates.sort()
        index = sorted_with_dates.index(date)
        sorted_paths.insert(index, fullpath)
    return sorted_paths, sorted_with_dates

def date_from_format(date, format, to_string_format=None):
    try:
        date = datetime.strptime(date, format)
    except:
        if len(date) > 19:
            fix_date = date[-19:]
            date = datetime.strptime(fix_date, format)
    if to_string_format:
        date_string = date.strftime(to_string_format)
        return date, date_string
    return date


def update_measure_and_datatype(data_object):
    if is_none(data_object.information.measure) or is_none(data_object.information.datatype):
        sensor_id = data_object.information.sensor_id
        data_id = data_object.information.data_id
        measure = None
        data_type = None
        if not data_object.information.datatype or data_object.information.measure:
            if sensor_id in Sensor.type_0:
                measure = Measures.WATIOS
                data_type = DataTypes.ELECTRIC_CLAMP
            elif sensor_id in Sensor.type_1:
                if data_id == 1:
                    measure = Measures.PRESENCE
                    data_type = DataTypes.PRESENCE
                elif data_id == 2:
                    measure = Measures.TEMPERATURE
                    data_type = DataTypes.TEMPERATURE
                elif data_id == 3:
                    measure = Measures.HUMIDITY
                    data_type = DataTypes.HUMIDITY
                elif data_id == 4:
                    measure = Measures.LUMINOSITY
                    data_type = DataTypes.LUMINOSITY
            if measure and data_type:
                data_object.information.set_info(measure=measure, datatype=data_type)

def update_data_objects(end_date=None, last_filepath=None, doids=None):
    """
        Check if there are data_objects in data_objects dictionary to delete.
    """
    to_delete = []
    if not doids:  # If None or empty, doids will be all doid.
        doids = data_objects.keys()
    for doid in doids:
        data_object = data_objects[doid]
        update_measure_and_datatype(data_object)
        if is_none(data_object.information.measure) or is_none(data_object.information.datatype) \
                or not data_object.is_valid():
            to_delete.append(doid)
            continue
        if last_filepath:
            data_object.information.set_info(file_path=last_filepath)
        if end_date:
            data_object.information.set_info(end_date=end_date)

    """
    for doid, data_object in data_objects.items():
        if doid in doids:  # "Doids" represents doids that must be processed
            if is_none(data_object.information.measure) or is_none(data_object.information.datatype):
                sensor_id = data_object.information.sensor_id
                data_id = data_object.information.data_id
                measure = None
                data_type = None
                if not data_object.information.datatype or data_object.information.measure:
                    if sensor_id in Sensor.type_0:
                        measure = Measures.WATIOS
                        data_type = DataTypes.ELECTRIC_CLAMP
                    elif sensor_id in Sensor.type_1:
                        if data_id == 1:
                            measure = Measures.PRESENCE
                            data_type = DataTypes.PRESENCE
                        elif data_id == 2:
                            measure = Measures.TEMPERATURE
                            data_type = DataTypes.TEMPERATURE
                        elif data_id == 3:
                            measure = Measures.HUMIDITY
                            data_type = DataTypes.HUMIDITY
                        elif data_id == 4:
                            measure = Measures.LUMINOSITY
                            data_type = DataTypes.LUMINOSITY
                if is_none(measure) or is_none(data_type) or not data_object.is_valid():
                    to_delete.append(doid)
                if measure and data_type:
                    data_object.information.set_info(measure=measure, datatype=data_type)
            if last_filepath and end_date:
                data_object.information.set_info(file_path=last_filepath, end_date=end_date)
    """
    for doid in set(to_delete):
        if doid in data_objects:
            del data_objects[doid]

def delete_unused_data_object():
    """
    Check if there are data_objects in data_objects dictionary to delete.
    """
    to_delete = []
    for doid, data_object in data_objects.items():
        if not data_object.x or not data_object.y:
            to_delete.append(doid)
    for doid in to_delete:
        del data_objects[doid]

class DataObjectReader():

    def __init__(self, path, sensor_type, load_data):
        self.path = path
        self.sensor_type = sensor_type
        self.load_data = load_data

    def read_refresh_data(self, i=None):
        global data_objects
        global saved_data_objects_doid
        global not_to_save_paths

        created_data_object = []

        pt("Refresing data...")

        sorted_paths = []
        sorted_with_dates = []
        for fullpath, root, name in get_files_from_path(paths=self.path, ends_in=".dat"):
            if name.split("_")[0] == str(self.sensor_type) and root == self.path:
                # Orders paths
                sorted_paths, sorted_with_dates = sort_paths_by_date_in_basename(sorted_paths,
                                                                                 sorted_with_dates,
                                                                                 fullpath,
                                                                                 name)
        if sorted_paths:
            not_to_save_paths.append(sorted_paths[-1])
        if self.sensor_type == 20:
            pt("")
        data_loaded_flag = self.load_historic_data(sorted_paths)
        # Info data
        first_date = None  # To see delta to show data
        end_date = None
        last_filepath = None
        total_files = len(sorted_paths)

        for file_count, file in enumerate(sorted_paths):
            pt("File " + str(file_count + 1) + " of " + str(total_files))
            file_sensor_id_date, extension_file = filename_and_extension_from_fullpath(fullpath=file)
            file_date = file_sensor_id_date.split(sep="_")[1]
            # Open file
            file_data = open(file, "r").read()
            lines = file_data.split("\n")
            total_lines = len(lines)
            pt("file", file)
            for line_count, line in enumerate(lines):
                if line_count % 5000 == 0 and line_count != 0:
                    gc.collect()
                printProgressBar(iteration=line_count, total=total_lines, prefix='File progress:', suffix='Complete',
                                 length=50)
                if len(line) > 1:
                    date_id_value = line.split(sep=";")
                    date, ID, value = date_id_value[0], int(date_id_value[1]), date_id_value[2]
                    date = date_from_format(date=date, format="%Y-%m-%d %H:%M:%S")
                    sensor_data_date_id = str(self.sensor_type) + "_" + date_id_value[1] + "_" + file_date
                    if ID not in Sensor.sensors_ids():  # sensors_ids must represent all different ids in all types of
                        # sensors
                        continue
                    if not sensor_data_date_id in data_objects:  # It means there is a new sensor data type.
                        information = InfoDataObject(sensor_id=self.sensor_type, start_date=date, data_id=ID,
                                                     file_path=file)
                        data_objects[sensor_data_date_id] = DataObject(information=information)
                        created_data_object.append(sensor_data_date_id)
                    data_object = data_objects[sensor_data_date_id]
                    if line_count == 0 or not first_date:
                        first_date = date
                    delta_minutes = (date - first_date).total_seconds()/60
                    if delta_minutes > 10. or data_loaded_flag:
                        value = float(value)
                        if self.sensor_type in Sensor.type_0:  # ElectricClamp
                            if ID == 1:
                                if len(date_id_value) > 3:
                                    try:
                                        clamp_1_value = value
                                        clamp_2_value = float(date_id_value[3])
                                        clamp_3_value = float(date_id_value[4])
                                        total_value = value + clamp_2_value + clamp_3_value
                                        multiple_y_values = [clamp_1_value, clamp_2_value, clamp_3_value, total_value]
                                        data_object.add(x=date, multiple_y_values=multiple_y_values)
                                    except Exception as e:
                                        pt("ERROR", e)
                        elif self.sensor_type in Sensor.global_sensor():  # Luminosity, presence, humidity and temperature
                            data_object.add(x=date, y=value)
                    if line_count + 1 == total_lines - 1:
                        end_date = date
                        last_filepath = file
            if end_date:
                update_data_objects(end_date=end_date, doids=created_data_object)
                created_data_object.clear()
                end_date = None
                last_filepath = None


    def load_historic_data(self, sorted_paths):
        """
        First, we get all saved files if exists. After that, we remove the appropiates files in "sorted_paths" to not
        read that file (and not load its data).

        Args:
            sorted_paths: Sorted paths with all paths with interested data.
        """
        global loaded_data_objects_doid

        data_loaded = False
        if load_data and sorted_paths:
            for fullpath, root, name in get_files_from_path(paths=checkpoint_path, ends_in=".npy"):
                sensor_id_data_id_date = name.split("_")
                sensor_id, data_id, date = sensor_id_data_id_date[0], sensor_id_data_id_date[1], \
                            datetime.strptime(sensor_id_data_id_date[2], "%Y%m%d")
                sensor_id_data_id_date_string = sensor_id + "_" + data_id + "_" + sensor_id_data_id_date[2]
                if int(sensor_id) == self.sensor_type:
                    for sorted_path in sorted_paths.copy():
                        info_sorted = os.path.splitext(os.path.basename(sorted_path))[0].split("_")
                        sensor_id_sorted = info_sorted[0]
                        if sensor_id_sorted == sensor_id:
                            date_sorted = datetime.strptime(info_sorted[1], "%Y%m%d")
                            if date_sorted == date:
                                sorted_paths.remove(sorted_path)
                    if sensor_id_data_id_date_string not in loaded_data_objects_doid:
                        pt("Loading data...")
                        data_objects[sensor_id_data_id_date_string] = DataObject().start_load(fullpath=fullpath)
                        loaded_data_objects_doid.append(sensor_id_data_id_date_string)
                        pt("Data loaded")
            if data_objects:
                data_loaded = True
        return data_loaded

def save_data_objects_checkpoint():
    """
    Args:
        filename: Filename to save
    """
    global saved_data_objects_doid
    pt("Saving data...")
    for doid, data_object in data_objects.items():
        file_to_check = data_object.information.file_path
        if not doid in saved_data_objects_doid and data_object.information.file_path not in not_to_save_paths:
            save_fullpath = checkpoint_path + doid + ".npy"
            if not file_exists_in_path_or_create_path(save_fullpath):
                data_object.start_save(fullpath=save_fullpath)
                saved_data_objects_doid.append(doid)

def histogram(vector, iteration=None, save=False, q=None):

    pt("q", q)
    pt("vector", vector)
    bins = np.delete(np.unique(vector), 0)
    pt("bins", bins)
    ret = plt.hist(vector, bins=bins)
    pt("ret", ret)
    pt("Max value count", max(ret[0]))
    index = np.argmax(ret[0])
    pt("Max value count numpy y", ret[0][index])
    pt("Max value index y", index)
    pt("list(ret[1])", list(ret[1]))
    pt("Max value x from index", ret[1][index])

    plt.title("Histogram")
    plt.ylabel("Normal value --> " + str(ret[1][index]))
    plt.grid(True)
    plt.show()

    return ret


def fourier(data, iteration=None, save=False):
    fft = np.fft.fft(data)
    frequencies = np.fft.fftfreq(data.shape[-1])
    pt("fft", fft)
    pt("frequencies", frequencies)
    pt("freqs", np.arange(256).shape)
    plt.plot(frequencies, fft.real, frequencies, fft.imag)
    plt.title("Fourier")
    #plt.ylabel("Normal value --> " + str(ret[1][index]))
    plt.show()


def log_transormation(data, iteration, save):
    pt("data", data[1])
    pt("data", type(data[1][0]))
    pt("data", np.log(data[1]))
    pt("data", type(np.log(data[1])))
    plt.plot(data[0], np.log(data[1]))
    plt.title("Log transformation")
    # plt.ylabel("Normal value --> " + str(ret[1][index]))
    plt.show()


def statistical_process(data, algorithm, save=False):
    """

    Args:
        data: data
        algorithm:
            - 1: Histogram

    Returns: data processed

    """
    # TODO (@gabvaztor) Get events from data. A event represent a patron in the data. After that, you can train
    # as supervised
    for i, type_data in enumerate(data):
        if i != 0 and i != 1:
            type_data = np.asarray(type_data)
            #histogram(type_data[1], iteration=i, save=save)
            #fourier(type_data[1], iteration=i, save=save)
            log_transormation(type_data, iteration=i, save=save)
    asfdaf
    return data

def calculate_datatypes():
    datatypes_number = 0
    for key, data_object in data_objects.items():
        datatypes_number += data_object.number_datatypes
    return datatypes_number

matplotlib = True
plotlylib = False
load_data = True
update_data(load_data=load_data)
save_data_objects_checkpoint()

def msg():
    big_dict = {}
    for doid, data_object in data_objects.items():
        big_dict[doid] = data_object.serialize()
    import msgpack
    with open(checkpoint_path + 'data.msgpack', 'w+') as outfile:
        msgpack.pack(big_dict, outfile)
msg()
#datatypes = calculate_datatypes() # Number of different data
#calculate_deltas_from_data()
#data = statistical_process(data=data, algorithm=1)

if plotlylib:
    for d in data:
        trace = dict(x=d[0], y=d[1])
        data_ = [trace]
        layout = dict(title='Time series with range slider and selectors')
        fig = dict(data=data_, layout=layout)
        plotly.plot(fig)

if matplotlib:  # Matplot lib
    pt("Creating and saving graphs...")
    actual_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for index, (doid, data_object) in enumerate(data_objects.items()):
        pt("DataObject " + str(index + 1) + " of " + str(len(data_objects)) + "...", data_object.information.datatype)
        for i in range(data_object.number_datatypes):
            pt("Creating graph " + str(i + 1) + " of " + str(data_object.number_datatypes) + "...", "")
            fig = plt.figure(figsize=(7, 7), dpi=160, facecolor='w', edgecolor='k')
            subplot = fig.add_subplot(1, 1, 1)
            left = 0.1  # the left side of the subplots of the figure
            right = 1.  # the right side of the subplots of the figure
            bottom = 0.05  # the bottom of the subplots of the figure
            top = 0.9  # the top of the subplots of the figure
            wspace = 0.2  # the amount of width reserved for space between subplots,
            # expressed as a fraction of the average axis width
            hspace = 0.2  # the amount of height reserved for space between subplots,
            # expressed as a fraction of the average axis height
            fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top,
                    wspace=wspace, hspace=hspace)
            add_to_graph(title=data_object.title)
            #figures.append(fig)
            #graphs.append(subplot)
            subplot.clear()

            x, y = data_object.pair_data(index=i)
            pt("x shape", x.shape)
            pt("y shape", y.shape)
            pt("Size of data_object in bytes", getsizeof(data_object))
            pt("Size of data_object.x in bytes", getsizeof(data_object.x))
            pt("Size of data_object.y in bytes", getsizeof(data_object.y))
            pt("Size of x in bytes", getsizeof(x))
            pt("Size of y in bytes", getsizeof(y))
            #total_bytes = data_object.total_bytes()
            if data_object.information.datatype == DataTypes.PRESENCE:
                s = [1]*len(x)
                plt.scatter(x, y, s=s)
            else:
                lines = subplot.plot(x, y)
                plt.setp(lines, linestyle='-', linewidth=.09, color='b')  # set both to dashed

            formatter = mdates.DateFormatter("%m/%d %H:%M:%S")
            subplot.set_ylabel(data_object.label)
            subplot.xaxis.set_major_formatter(formatter)
            fig.autofmt_xdate()
            # Save graph
            save_path_graph = save_path + "Graphs\\" + actual_time + "\\" + \
                              data_object.information.datatype + "(" + str(data_object.information.sensor_id) + "_" + \
                              str(data_object.information.data_id) + ")" + ".png"
            save_path_graph = check_file_exists_and_change_name(save_path_graph)
            pt("path_save", save_path_graph)
            fig.savefig(fname=save_path_graph, dpi=1800)
            gc.collect()
    #animated_clamp = animation.FuncAnimation(figures[0], clamp_function, interval=1000000000)
    #animated_global_sensor = animation.FuncAnimation(figures[1], global_sensor_function, interval=10000000)

