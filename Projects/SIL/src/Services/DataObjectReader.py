import time

from Projects.SIL.src.DTO.ElectricClamp import ElectricClamp
from Projects.SIL.src.DTO.Humidity import Humidity
from Projects.SIL.src.DTO.Luminosity import Luminosity
from Projects.SIL.src.DTO.Presence import Presence
from Projects.SIL.src.DTO.Temperature import Temperature
from UsefulTools.UtilsFunctions import pt, printProgressBar, get_files_from_path, create_directory_from_fullpath, \
    file_exists_in_path_or_create_path, is_none, check_file_exists_and_change_name, \
    filename_and_extension_from_fullpath, transform_to_list
from AsynchronousThreading import execute_asynchronous_thread
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import matplotlib.dates as mdates
from datetime import datetime, date as typedate, timedelta
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
style_ = style_list[19]
pt(style_)
style.use(style=style_)

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

def update_data(load_data, dates_to_load=None, update_all_data=True, read_data=True, sensor_types=None,
                cores_ids=None, data_ids=None):
    sensor_types = []
    #DataObjectReader(path=miranda_path, sensor_type=20, load_data=load_data).read_refresh_data()

    if not dates_to_load:
        dates_to_load = []
    if update_all_data:
        sensor_types = get_all_sensor_types(actual_sensor_types=sensor_types, path=miranda_path)
    else:
        sensor_types = []
    sensor_types.remove(20)
    for core_id in sensor_types.copy():
        if core_id not in cores_ids and len(cores_ids) > 0:
            sensor_types.remove(core_id)
    for sensor_type in sensor_types:
        DataObjectReader(path=miranda_path, sensor_type=sensor_type,
                         load_data=load_data, load_dates=dates_to_load,
                         read_data=read_data, save_data=True,
                         cores_ids=cores_ids, data_ids=data_ids).read_refresh_data()
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
        try:
            if len(date) > 19:
                fix_date = date[-19:]
                date = datetime.strptime(fix_date, format)
        except:
            traceback.print_exc()
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


def filter_paths(paths, data_ids, cores_ids, load_dates):
    if not data_ids:
        data_ids = []
    if not cores_ids:
        cores_ids = []
    for sorted_path in paths.copy():
        info_sorted = os.path.splitext(os.path.basename(sorted_path))[0].split("_")
        sensor_id_sorted = int(info_sorted[0])
        if cores_ids:
            if sensor_id_sorted not in cores_ids:
                paths.remove(sorted_path)
                continue
            elif load_dates:
                end_date = load_dates[1]
                date_sorted = datetime.strptime(info_sorted[1], "%Y%m%d")
                if date_sorted <= end_date and sorted_path in paths:
                    paths.remove(sorted_path)



class DataObjectReader():

    def __init__(self, path, sensor_type, load_data, save_data, read_data, load_dates, cores_ids, data_ids):
        self.path = path
        self.sensor_type = sensor_type
        self.load_data = load_data
        self.save_data = save_data
        self.read_data = read_data
        if not load_dates:
            load_dates = []
        self.load_dates = load_dates
        if not cores_ids:
            cores_ids = []
        self.cores_ids = cores_ids
        if not data_ids:
            data_ids = []
        self.data_ids = data_ids

    def read_refresh_data(self, i=None):
        global data_objects
        global saved_data_objects_doid
        global not_to_save_paths

        created_data_object = []

        pt("Refresing data [Core_ID:" + str(self.sensor_type) + "]...")

        sorted_paths = []
        sorted_with_dates = []
        for fullpath, root, name in get_files_from_path(paths=self.path, ends_in=".dat"):
            if name.split("_")[0] == str(self.sensor_type) and root == self.path:
                # Orders paths
                sorted_paths, sorted_with_dates = sort_paths_by_date_in_basename(sorted_paths,
                                                                                 sorted_with_dates,
                                                                                 fullpath,
                                                                                 name)
        if not self.save_data:
            not_to_save_paths = sorted_paths
        elif sorted_paths:
            not_to_save_paths.append(sorted_paths[-1])
        data_loaded_flag = self.load_historic_data(sorted_paths, load_dates=self.load_dates, delete_higher_dates=False,
                                                   data_ids=self.data_ids, cores_ids=self.cores_ids)
        if not self.read_data:
            sorted_paths.clear()
        filter_paths(paths=sorted_paths, data_ids=self.data_ids, cores_ids=self.cores_ids, load_dates=self.load_dates)

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
                    date, ID, value = date_id_value[0], date_id_value[1], date_id_value[2]
                    sensor_data_date_id = str(self.sensor_type) + "_" + ID + "_" + file_date
                    try:
                        ID = int(ID)
                        value = float(value)
                        date = date_from_format(date=date, format="%Y-%m-%d %H:%M:%S")
                    except:
                        traceback.print_exc()
                        pt("Line count", line_count)
                        pt("Line", line)
                        pt("File", file)
                        pt("Unique ID", sensor_data_date_id)
                        continue
                    if ID not in Sensor.sensors_ids() or (ID not in data_ids and data_ids):  # sensors_ids must
                        # represent all different ids in all types of sensors
                        continue
                    if not sensor_data_date_id in data_objects:  # It means there is a new sensor data type.
                        information = InfoDataObject(sensor_id=self.sensor_type, start_date=date, data_id=ID,
                                                     file_path=file, unique_id=sensor_data_date_id)
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


    def load_historic_data(self, sorted_paths, load_dates=None, delete_higher_dates=False, data_ids=None,
                           cores_ids=None):
        """
        First, we get all saved files if exists. After that, we remove the appropiates files in "sorted_paths" to not
        read that file (and not load its data).

        Args:
            sorted_paths: Sorted paths with all paths with interested data.
        """
        global loaded_data_objects_doid
        global saved_data_objects_doid

        if not data_ids:
            data_ids = []
        if not cores_ids:
            cores_ids = []

        start_date = None
        end_date = None
        data_loaded = False
        if load_data:
            if load_dates:
                start_date = load_dates[0]
                end_date = load_dates[-1]
            for fullpath, root, name in get_files_from_path(paths=checkpoint_path, ends_in=".npy"):
                sensor_id_data_id_date = name.split("_")
                sensor_id, data_id, date = sensor_id_data_id_date[0], sensor_id_data_id_date[1], \
                            datetime.strptime(sensor_id_data_id_date[2], "%Y%m%d")
                sensor_id_data_id_date_string = sensor_id + "_" + data_id + "_" + sensor_id_data_id_date[2]
                load_flag = False
                meets_core_flag = ((int(sensor_id) in cores_ids) or (not cores_ids))
                meets_data_flag = ((int(data_id) in data_ids) or (not data_ids))
                if load_dates:
                    if date >= start_date and date <= end_date and int(sensor_id) == self.sensor_type \
                            and meets_core_flag and meets_data_flag:
                        load_flag = True
                elif int(sensor_id) == self.sensor_type and meets_core_flag and meets_data_flag:
                    load_flag = True
                if load_flag:
                    for sorted_path in sorted_paths.copy():
                        info_sorted = os.path.splitext(os.path.basename(sorted_path))[0].split("_")
                        sensor_id_sorted = info_sorted[0]
                        if sensor_id_sorted == sensor_id:
                            date_sorted = datetime.strptime(info_sorted[1], "%Y%m%d")
                            if date_sorted == date:
                                sorted_paths.remove(sorted_path)
                            if date_sorted < start_date:
                                sorted_paths.remove(sorted_path)
                            if delete_higher_dates:
                                if date_sorted > end_date:
                                    sorted_paths.remove(sorted_path)
                    if sensor_id_data_id_date_string not in loaded_data_objects_doid:
                        pt("Loading data...")
                        data_objects[sensor_id_data_id_date_string] = DataObject().start_load(fullpath=fullpath)
                        loaded_data_objects_doid.append(sensor_id_data_id_date_string)
                        data_loaded = True
                        pt("Data with id [" + sensor_id_data_id_date_string + "] loaded")

        return data_loaded

def save_data_objects_checkpoint(save_data=True):
    """
    Args:
        save_data: Filename to save
    """
    global saved_data_objects_doid
    for doid, data_object in data_objects.items():
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


def msg():
    big_dict = {}
    for doid, data_object in data_objects.items():
        big_dict[doid] = data_object.serialize()
    import msgpack
    actual_time = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    try:
        with open(checkpoint_path + actual_time + '_data.msgpack', 'wb') as outfile:
            msgpack.pack(big_dict, outfile)
    except:
        pass

    import pickle
    pickle.dump(big_dict, open(checkpoint_path + actual_time + "_picke_data.p", "wb"))
    del big_dict

def sort_data_objects():
    global data_objects
    new_data_objects = {}
    for key, value in sorted(data_objects.items()):
        new_data_objects[key] = value
    data_objects = new_data_objects
    del new_data_objects

def grahps_process():
    if plotlylib:
        for d in data:
            trace = dict(x=d[0], y=d[1])
            data_ = [trace]
            layout = dict(title='Time series with range slider and selectors')
            fig = dict(data=data_, layout=layout)
            plotly.plot(fig)

    if matplotlib:  # Matplot lib
        pt("Creating and saving graphs...")
        actual_time = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
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
                    pass
                    #s = [1]*len(x)
                    #plt.scatter(x, y, s=s)
                else:
                    lines = subplot.plot(x, y)
                    plt.setp(lines, linestyle=':', linewidth=.2, color='red')  # set both to dashed

                    formatter = mdates.DateFormatter("%m/%d %H:%M:%S")
                    subplot.set_ylabel(data_object.label)
                    subplot.xaxis.set_major_formatter(formatter)
                    fig.autofmt_xdate()
                    # Save graph
                    save_path_graph = save_path + "Graphs\\" + actual_time + "\\" + \
                                      data_object.information.datatype + "(" + data_object.unique_doid_date+ ")" + ".png"
                    save_path_graph = check_file_exists_and_change_name(save_path_graph, char="_")
                    pt("path_save", save_path_graph)
                    fig.savefig(fname=save_path_graph, dpi=1000)
                    gc.collect()
        #animated_clamp = animation.FuncAnimation(figures[0], clamp_function, interval=1000000000)
        #animated_global_sensor = animation.FuncAnimation(figures[1], global_sensor_function, interval=10000000)

def save_data_object_graph(path, fig, data_object, actual_time, format=".png"):
    # Save graph
    save_path_graph = path + "Graphs\\" + actual_time + "\\" + \
                      data_object.information.datatype + "(" + data_object.unique_doid_date + ")" + format
    save_path_graph = check_file_exists_and_change_name(save_path_graph, char="_")
    pt("path_save", save_path_graph)
    fig.savefig(fname=save_path_graph, dpi=1000)


def filter_changes_list(changes, values, top_limit=None, bottom_limit=None):
    """

    Args:
        changes:
        values:
        top_limit:
        bottom_limit:

    Returns:

    """
    changes = transform_to_list(changes)
    values = transform_to_list(values)
    for change_index in changes.copy():
        value = values[change_index]
        if not is_none(top_limit) and not is_none(bottom_limit):
            top_limit = transform_to_list(top_limit)
            bottom_limit = transform_to_list(bottom_limit)
            top_value = top_limit[change_index]
            bottom_value = bottom_limit[change_index]
            if value > bottom_value and value < top_value:
                changes.remove(change_index)
        else:
            break
    return changes

def data_analysis(cores_ids=None, data_ids=None, join_data=False):

    if not cores_ids:
        cores_ids = []
    if not data_ids:
        data_ids = []

    joined_data_objects = {}

    if cores_ids and data_ids and join_data:  #Join data
        for unique_id, data_object in data_objects.items():
            if data_object.information.data_id in data_ids and data_object.information.sensor_id in cores_ids:
                if not data_object.unique_doid in joined_data_objects:
                    joined_data_objects[data_object.unique_doid] = data_object
                else:
                    joined_data_objects[data_object.unique_doid].join_data_object(data_object)

    def filter_dict_by_id(dictionary, cores_ids=None, data_ids=None):

        to_delete = []

        if not cores_ids:
            cores_ids = []
        if not data_ids:
            data_ids = []

        if cores_ids or data_ids:
            for unique_id, data_object in dictionary.items():
                core_id, data_id = int(unique_id.split("_")[0]), int(unique_id.split("_")[1])
                if core_id not in cores_ids and data_id not in data_ids:
                    to_delete.append(unique_id)
                else:
                    if core_id not in cores_ids:
                        to_delete.append(unique_id)
                    elif data_id not in data_ids:
                        to_delete.append(unique_id)

        # Delete unnecessary data_objects
        for unique_id in to_delete:
            del dictionary[unique_id]

        return dictionary

    if not joined_data_objects:
        joined_data_objects = data_objects

    joined_data_objects = filter_dict_by_id(dictionary=joined_data_objects, cores_ids=cores_ids, data_ids=data_ids)
    actual_time = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")

    for unique_doid, data_object in joined_data_objects.items():
        pt("Generating PDF of [" + str(unique_doid) + "]...")
        datatype = data_object.information.datatype
        data_object_frame = data_object.dataframe()

        from scipy.signal import savgol_filter
        data_object_frame["Filtered"] = savgol_filter(data_object_frame[datatype], window_length=199, polyorder=3)
        data_object_frame["Interpolated"] = savgol_filter(data_object_frame[datatype], window_length=99, polyorder=3, deriv=2, delta=.5)
        data_object_frame["Filtered_Interpolated"] = savgol_filter(data_object_frame[datatype], window_length=199, polyorder=3, deriv=2, delta=0.5)
        data_object_frame["savgol_filter"] = savgol_filter(data_object_frame[datatype], window_length=3, polyorder=2, deriv=2)

        import random
        wind = random.randint(10, 300)
        wind = 6
        sigma = random.uniform(0.7, 3)
        sigma = 0.5
        std = data_object_frame.std()[datatype]
        pt("window", wind)
        pt("sigma", sigma)
        pt("std", std)

        from matplotlib.backends.backend_pdf import PdfPages
        pdf_path = output_path + actual_time + data_object.information.datatype + "(" \
                   + data_object.unique_doid_date + ")" + ".pdf"
        n = 0
        fig = None
        type_graphs = [""] * 24
        pdf = PdfPages(pdf_path)
        for i, element in enumerate(type_graphs):

            #plot_num = 321
            #plt.subplot(plot_num)
            fig = plt.figure(figsize=(10, 10), dpi=1000)
            fig.clf()
            fig.text(4.25 / 8.5, 0.5 / 11., str(i + 1), ha='center', fontsize=8)
            text = "Windows Size:" + str(wind) + " | Sigma:" + str(sigma) + " | Std:" + "{0:.2f}".format(std)
            #plt.text(0.2, 0.95, text, transform=fig.transFigure, size=16)
            fig.suptitle(text, fontsize=14, fontweight='bold')
            if i == 0:
                plt.title("Histogram|" + data_object.title)
                ax = plt.gca()
                data_object_frame.hist(ax=ax)
                plt.grid(True)
            elif i == 1:
                plt.title("Floor and Ceiling|" + data_object.title)
                data_object_frame["Floor"] = data_object_frame[datatype].rolling(window=wind).mean() - \
                                             (sigma * data_object_frame[datatype].rolling(window=wind).std())
                data_object_frame["Ceiling"] = data_object_frame[datatype].rolling(window=wind).mean() + \
                                               (sigma * data_object_frame[datatype].rolling(window=wind).std())
                #data_object_frame.cumsum(axis=1)
                plt.plot(data_object_frame.index.values, data_object_frame[datatype], label=datatype, linewidth=0.2)
                plt.plot(data_object_frame.index.values, data_object_frame["Floor"], label="Floor", linewidth=0.05)
                plt.plot(data_object_frame.index.values, data_object_frame["Ceiling"], label="Ceiling", linewidth=0.05)
                plt.grid(True)
                plt.gca().legend()
            elif i == 2:
                plt.title("Anomalies|" + data_object.title)
                data_object_frame["Anomaly"] = data_object_frame.apply(
                    lambda row: row[datatype] if (
                            row[datatype] <= row["Floor"] or row[datatype] >= row["Ceiling"]) else None
                    , axis=1)

                s = np.full(len(data_object_frame[datatype]), 0.05)
                plt.scatter(data_object_frame.index.values, data_object_frame["Anomaly"], s=s, color="r",
                            label="Anomalies")
                plt.plot(data_object_frame.index.values, data_object_frame[datatype], label=datatype, linewidth=0.05)
                plt.grid(False)
                plt.gca().legend()
            elif i == 6:
                plt.title("Filtered|" + data_object.title)
                plt.plot(data_object_frame.index.values, data_object_frame["Filtered"], label=datatype, linewidth=0.05)
                plt.gca().legend()
            elif i == 7:
                plt.title("Interpolated|" + data_object.title)
                plt.plot(data_object_frame.index.values, data_object_frame["Interpolated"], label=datatype, linewidth=0.05)
                plt.gca().legend()
            elif i == 8:
                plt.title("Filtered_Interpolated|" + data_object.title)
                plt.plot(data_object_frame.index.values, data_object_frame["Filtered_Interpolated"], label=datatype, linewidth=0.05)
                plt.gca().legend()
            elif i == 9:
                plt.title("Events|" + data_object.title)
                window = 101
                #der2 = savgol_filter(data_object_frame[datatype], window_length=3, polyorder=2, deriv=2)
                der2 = data_object_frame["savgol_filter"]
                max_der2 = np.max(np.abs(der2))
                large = np.where(np.abs(der2) > max_der2 / 4)[0]
                gaps = np.diff(large) > window
                begins = np.insert(large[1:][gaps], 0, large[0])
                ends = np.append(large[:-1][gaps], large[-1])
                changes = list(((begins + ends) / 2).astype(np.int))
                pt("changes", changes)

                index_series = 0

                def value_update_i(value, list, top_limit=None, bottom_limit=None):
                    nonlocal index_series
                    value_return = None
                    index = -1
                    if index_series in list:
                        index = list.index(index_series)
                        value_return = value
                        if top_limit and bottom_limit:
                            if value_return > bottom_limit and value_return < top_limit:
                                value_return = None
                    # Check if last element
                    if len(list) - 1 == index:
                        index_series = 0
                    else:
                        index_series += 1
                    return value_return

                data_object_frame["Events"] = data_object_frame.apply(
                    lambda row: value_update_i(value=row[datatype], list=changes), axis=1)
                pt("index", index_series)
                index_series = 0
                plt.plot(data_object_frame.index.values, data_object_frame[datatype])
                plt.plot(data_object_frame.index.values, data_object_frame["Events"], 'ro')
                plt.gca().legend()
            elif i == 10:
                s = np.asarray(data_object_frame[datatype].tolist())
                a_sign = np.sign(s)
                sign_change = ((np.roll(a_sign, 1) - a_sign) != 0).astype(int)
                changes = list(np.nonzero(sign_change == 1))
                data_object_frame["Events2"] = data_object_frame.apply(
                    lambda row: value_update_i(value=row[datatype], list=changes), axis=1)
                index_series = 0
                pt("changes2", changes)
                plt.plot(data_object_frame.index.values, data_object_frame[datatype])
                plt.plot(data_object_frame.index.values, data_object_frame["Events2"], 'ro')
                plt.gca().legend()
            elif i == 11:
                plt.plot(data_object_frame.index.values, data_object_frame["savgol_filter"])
                plt.gca().legend()
            elif i == 12:
                percent = 1.2
                percent2 = 0.8
                mean = data_object_frame["Filtered_Interpolated"].median(axis=0)
                lent = len(data_object_frame.index)
                data_object_frame["Floor_savgol_filter"] = np.full(shape=lent, fill_value=mean - (sigma * data_object_frame["Filtered_Interpolated"].std()))
                data_object_frame["Ceiling_savgol_filter"] = np.full(shape=lent, fill_value=mean + (sigma * data_object_frame["Filtered_Interpolated"].std()))
                data_object_frame["Mean_savgol_filter"] = np.full(shape=lent, fill_value=mean)
                data_object_frame["Anomaly_savgol_filter"] = data_object_frame.apply(
                    lambda row: row["Filtered_Interpolated"] if (
                            row["Filtered_Interpolated"] <= row["Floor_savgol_filter"]
                            or row["Filtered_Interpolated"] >= row["Ceiling_savgol_filter"]) else None
                    , axis=1)
                #plt.plot(data_object_frame.index.values, data_object_frame[datatype])
                plt.plot(data_object_frame.index.values, data_object_frame["Filtered_Interpolated"])
                plt.plot(data_object_frame.index.values, data_object_frame["Floor_savgol_filter"])
                plt.plot(data_object_frame.index.values, data_object_frame["Ceiling_savgol_filter"])
                plt.plot(data_object_frame.index.values, data_object_frame["Mean_savgol_filter"], "black")
                plt.plot(data_object_frame.index.values, data_object_frame["Anomaly_savgol_filter"], 'ro')
                plt.gca().legend()
            elif i == 13:
                #plt.plot(data_object_frame.index.values, data_object_frame["Filtered_Interpolated"].resample(), style=':')
                #plt.plot(data_object_frame.index.values, data_object_frame["Filtered_Interpolated"].asfreq(), style='--')
                plt.plot(data_object_frame.index.values, data_object_frame["Filtered_Interpolated"], alpha=0.5)
                plt.gca().legend()
            elif i == 14:
                plt.plot(data_object_frame.index.values, data_object_frame[datatype].pct_change(), alpha=0.5)
                plt.gca().legend()
            elif i == 15:
                plt.plot(data_object_frame.index.values, data_object_frame[datatype].pct_change(periods=30), alpha=0.7)
                plt.gca().legend()
            elif i == 16:
                g = np.gradient(data_object_frame["Filtered_Interpolated"])
                data_object_frame["Gradients_Filtered_Interpolated"] = np.gradient(data_object_frame["Filtered_Interpolated"])
                plt.plot(data_object_frame.index.values, data_object_frame["Gradients_Filtered_Interpolated"])
                plt.gca().legend()
            elif i == 17:
                g = np.gradient(data_object_frame[datatype])
                data_object_frame["Gradients"] = np.gradient(data_object_frame[datatype])
                plt.plot(data_object_frame.index.values, data_object_frame["Gradients"])
                plt.gca().legend()
            elif i == 18:
                data_object_frame["Gradients_Filtered_Interpolated"] = np.gradient(data_object_frame["Filtered_Interpolated"])
                sign = np.sign(data_object_frame["Gradients_Filtered_Interpolated"])
                sign_change = ((np.roll(sign, 1) - sign) != 0).astype(int)
                changes = list(np.nonzero(sign_change == 1)[0])
                data_object_frame["Events2"] = data_object_frame.apply(
                    lambda row: value_update_i(value=row["Filtered_Interpolated"], list=changes), axis=1)
                index_series = 0
                plt.plot(data_object_frame.index.values, data_object_frame["Filtered_Interpolated"])
                plt.plot(data_object_frame.index.values, data_object_frame["Events2"], "ro")
                plt.gca().legend()
            elif i == 19:
                data_object_frame["Gradients_Filtered_Interpolated"] = np.gradient(data_object_frame["Filtered_Interpolated"])
                sign = np.sign(data_object_frame["Gradients_Filtered_Interpolated"])
                sign_change = ((np.roll(sign, 1) - sign) != 0).astype(int)
                changes = list(np.nonzero(sign_change == 1)[0])

                mean = data_object_frame["Filtered_Interpolated"].median(axis=0)
                lent = len(data_object_frame.index)
                data_object_frame["Floor_savgol_filter"] = np.full(shape=lent, fill_value=mean - (
                        sigma * data_object_frame["Filtered_Interpolated"].std()))
                data_object_frame["Ceiling_savgol_filter"] = np.full(shape=lent, fill_value=mean + (
                        sigma * data_object_frame["Filtered_Interpolated"].std()))
                data_object_frame["Events2"] = data_object_frame.apply(
                    lambda row: value_update_i(value=row["Filtered_Interpolated"], list=changes,
                                               top_limit=row["Ceiling_savgol_filter"],
                                               bottom_limit=row["Floor_savgol_filter"]), axis=1)

                index_series = 0
                plt.plot(data_object_frame.index.values, data_object_frame["Filtered_Interpolated"])
                plt.plot(data_object_frame.index.values, data_object_frame["Floor_savgol_filter"])
                plt.plot(data_object_frame.index.values, data_object_frame["Ceiling_savgol_filter"])
                plt.plot(data_object_frame.index.values, data_object_frame["Events2"], "ro")
                plt.gca().legend()
            elif i == 20:
                data_object_frame["Gradients_Filtered_Interpolated"] = np.gradient(data_object_frame["Filtered_Interpolated"])
                sign = np.sign(data_object_frame["Gradients_Filtered_Interpolated"])
                sign_change = ((np.roll(sign, 1) - sign) != 0).astype(int)
                changes = list(np.nonzero(sign_change == 1)[0])
                data_object_frame["Events_1"] = data_object_frame.apply(
                    lambda row: value_update_i(value=row["Filtered"], list=changes), axis=1)
                index_series = 0
                plt.plot(data_object_frame.index.values, data_object_frame["Filtered"])
                plt.plot(data_object_frame.index.values, data_object_frame["Events_1"], "ro")
                plt.gca().legend()
            elif i == 21:
                data_object_frame["Gradients_Filtered_Interpolated"] = np.gradient(data_object_frame["Filtered_Interpolated"])
                mean = data_object_frame["Filtered_Interpolated"].median(axis=0)
                lent = len(data_object_frame.index)
                data_object_frame["Floor_savgol_filter"] = np.full(shape=lent, fill_value=mean - (
                        sigma * data_object_frame["Filtered_Interpolated"].std()))
                data_object_frame["Ceiling_savgol_filter"] = np.full(shape=lent, fill_value=mean + (
                            sigma * data_object_frame["Filtered_Interpolated"].std()))
                sign = np.sign(data_object_frame["Gradients_Filtered_Interpolated"])
                sign_change = ((np.roll(sign, 1) - sign) != 0).astype(int)
                changes = list(np.nonzero(sign_change == 1)[0])
                changes = filter_changes_list(changes=changes, values=data_object_frame["Filtered_Interpolated"],
                                              top_limit=data_object_frame["Ceiling_savgol_filter"],
                                              bottom_limit=data_object_frame["Floor_savgol_filter"])
                data_object_frame["Events_1"] = data_object_frame.apply(
                    lambda row: value_update_i(value=row["Filtered"], list=changes), axis=1)
                index_series = 0
                plt.plot(data_object_frame.index.values, data_object_frame["Filtered"], )
                plt.plot(data_object_frame.index.values, data_object_frame[datatype], "green", alpha=0.5)
                plt.plot(data_object_frame.index.values, data_object_frame["Events_1"], "ro")
                plt.gca().legend()
            elif i == 22:
                # TODO (@gabvaztor) Do i=18 to raw data. Check events and filter with gradient and % of change.
                data_object_frame["Gradients_Raw"] = np.gradient(
                    data_object_frame[datatype])
                sign = np.sign(data_object_frame["Gradients_Raw"])
                sign_change = ((np.roll(sign, 1) - sign) != 0).astype(int)
                changes = list(np.nonzero(sign_change == 1)[0])
                data_object_frame["Events3"] = data_object_frame.apply(
                    lambda row: value_update_i(value=row[datatype], list=changes), axis=1)
                index_series = 0
                plt.plot(data_object_frame.index.values, data_object_frame[datatype], "blue")
                plt.plot(data_object_frame.index.values, data_object_frame["Filtered"], "green", alpha=0.5)
                plt.plot(data_object_frame.index.values, data_object_frame["Events3"], "ro")
                plt.gca().legend()
            elif i == 23:
                data_object_frame["Gradients_Raw"] = np.gradient(
                    data_object_frame[datatype])

                mean = data_object_frame[datatype].median(axis=0)
                lent = len(data_object_frame.index)
                std_ = data_object_frame[datatype].std()
                # TODO (@gabvaztor) Change floor and ceiling to be a curve across datatype
                floor_raw = np.full(shape=lent, fill_value=(mean - (sigma * std_)))
                ceiling_raw = np.full(shape=lent, fill_value=(mean + (sigma * std_)))

                sign = np.sign(data_object_frame["Gradients_Raw"])
                sign_change = ((np.roll(sign, 1) - sign) != 0).astype(int)
                changes = list(np.nonzero(sign_change == 1)[0])
                changes = filter_changes_list(changes=changes, values=data_object_frame[datatype],
                                              top_limit=ceiling_raw,
                                              bottom_limit=floor_raw)
                data_object_frame["Events4"] = data_object_frame.apply(
                    lambda row: value_update_i(value=row[datatype], list=changes), axis=1)
                index_series = 0
                plt.plot(data_object_frame.index.values, data_object_frame[datatype], "blue")
                plt.plot(data_object_frame.index.values, data_object_frame["Filtered"], "green", alpha=0.5)
                plt.plot(data_object_frame.index.values, data_object_frame["Events4"], "ro")
                plt.plot(data_object_frame.index.values, ceiling_raw, label="Ceiling_raw")
                plt.plot(data_object_frame.index.values, floor_raw, label="Floor_raw")
                plt.gca().legend()
            else:
                fig, ax = plt.subplots()
                fig.patch.set_visible(False)
                ax.axis('off')
                fig.set_size_inches(8.3, 11.7)
                plt.text(4.25 / 8.5, 0.5 / 11., str(i + 1), ha='center', fontsize=8)
                if i == 3:
                    txt = "Correlations"
                    plt.text(0.5, 0.9, txt, horizontalalignment='center', verticalalignment='center',
                             transform=fig.transFigure, size=16, color='blue', style='italic')
                    string_correlation = str(data_object_frame.corr())
                    plt.text(0.5, 0.55, string_correlation, horizontalalignment='center', verticalalignment='center',
                             transform=fig.transFigure, size=16)
                    #plt.table()
                elif i == 4:
                    fig.patch.set_visible(False)
                    txt = "Summary"
                    plt.text(0.5, 0.9, txt, horizontalalignment='center', verticalalignment='center',
                             transform=fig.transFigure, size=16, color='blue', style='italic')
                    string_summary = str(data_object_frame.describe())
                    plt.text(0.5, 0.5, string_summary, horizontalalignment='center', verticalalignment='center',
                             transform=fig.transFigure, size=13)
                elif i == 5:
                    import statsmodels.api as sm
                    # pip3 install statsmodels
                    # pip3 install patsy
                    fig.patch.set_visible(False)
                    seconds = data_object.seconds_array()
                    mod = sm.OLS(seconds, data_object_frame[datatype])
                    res = mod.fit()
                    txt = "Stats Model Summary"
                    plt.text(0.5, 0.9, txt, horizontalalignment='center', verticalalignment='center',
                             transform=fig.transFigure, size=16, color='blue', style='italic')
                    string_summary = str(res.summary())
                    plt.text(0.5, 0.6, string_summary, horizontalalignment='center', verticalalignment='center',
                             transform=fig.transFigure, size=13)

            #fig.autofmt_xdate()
            pdf.savefig(fig, transparent=True)
        # formatter = mdates.DateFormatter("%m/%d %H:%M:%S")
        pt("PDF of data_object [" + unique_doid + "] saved")
        fig.clear()
        plt.close("all")
        pdf.close()
        gc.collect()
        """
        save_data_object_graph(path=output_path, data_object=data_object, actual_time=actual_time,
                               fig=data_object_frame.plot()[0][0])
                               
        pt()
        """



if __name__ == '__main__':

    # ##### #
    # PATHS #
    # ##### #
    miranda_path = "\\\\192.168.1.220\\miranda\\"
    miranda_path = "..\\..\\data\\temp\\"
    # miranda_path = "F:\\Data_Science\\Projects\\Smartiotlabs\\Data\\"
    # Save image path
    save_path = "E:\\SmartIotLabs\\AI_Department\\Data\\Sensors\\"
    checkpoint_path = "E:\\SmartIotLabs\\AI_Department\\Data\\Sensors\\Checkpoints\\"
    checkpoint_path = "..\\..\\data\\intermediate\\"
    output_path = "..\\..\\results\\output\\PDFs\\"

    # ######### #
    # VARIABLES #
    # ######### #
    matplotlib = True
    plotlylib = False
    read_data = False
    load_data = True
    save_data = True
    load_dates = True
    update_all_data = True
    join_data = False

    start_date_to_load = datetime(year=2018, month=7, day=25)
    end_date_to_load = datetime(year=2018, month=7, day=25) + timedelta(days=1) - timedelta(seconds=1)
    #end_date_to_load = datetime(year=2018, month=7, day=31) + timedelta(days=1) - timedelta(seconds=1)
    dates_to_load = [start_date_to_load, end_date_to_load]
    if not load_dates:
        dates_to_load.clear()

    cores_ids = [1]
    data_ids = [2]

    # ############ #
    # MAIN PROCESS #
    # ############ #
    update_data(load_data=load_data, dates_to_load=dates_to_load, update_all_data=update_all_data, read_data=read_data,
                cores_ids=cores_ids, data_ids=data_ids)
    sort_data_objects()
    save_data_objects_checkpoint(save_data=save_data)

    data_analysis(cores_ids=cores_ids, data_ids=data_ids, join_data=join_data)
    #grahps_process()
    # msg()
    # data = statistical_process(data=data, algorithm=1)
    pt("Run timestamp", datetime.now())



