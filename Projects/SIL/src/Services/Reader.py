import time

from Projects.SIL.src.DTO.ElectricClamp import ElectricClamp
from Projects.SIL.src.DTO.Humidity import Humidity
from Projects.SIL.src.DTO.Luminosity import Luminosity
from Projects.SIL.src.DTO.Presence import Presence
from Projects.SIL.src.DTO.Temperature import Temperature
from UsefulTools.UtilsFunctions import pt, printProgressBar, get_files_from_path, create_directory_from_fullpath
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
from sys import getsizeof
from Projects.SIL.src.Services.DataObject import DataObject, DataTypes, InfoDataObject, Measures





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
miranda_path = "\\\\192.168.1.38\\miranda\\"
#miranda_path = "F:\\Data_Science\\Projects\\Smartiotlabs\\Data\\"
# global_sensor_filepath="E:\\SmartIotLabs\\DATA\\nEW\\1_20180702.dat"
# Save image path
save_path = "E:\\SmartIotLabs\\AI_Department\\Data\\Sensors\\"
checkpoint_path = "E:\\SmartIotLabs\\AI_Department\\Data\\Sensors\\Checkpoints\\"

#figures = []
#graphs = []

# Info
titles = ["Power disaggregation 1", "Power disaggregation 2", "Power disaggregation 3", "Power disaggregation total",
          "Temperature",
          "Humidity",
          "Luminosity",
          "Presence"
          ]
ylabels = ["Power (W)", "Power (W)", "Power (W)", "Power (W)",
           "Cº",
           "%",
           "Lux",
           "True / False"]

# Data
clamp_date_1 = []
clamp_date_2 = []
clamp_date_3 = []
presence_date = []
humidity_date = []
luminosity_date = []
temperature_date = []

clamp_value_1 = []
clamp_value_2 = []
clamp_value_3 = []
presence_value = []
humidity_value = []
luminosity_value = []
temperature_value = []

data_global = []
delta_data = []

# Information
sensor_data_ids = []  # Contains the information of sensors and data_ids. The format is: ["sensorID_dataID", ...]
# Will be used to check when a type data was used or not.
data_objects = {}

def add_to_graph(title=None, ylabel=None, ax=None):
    plt.title(title)
    plt.suptitle(title)
    plt.grid()

def calculate_deltas_from_data():
    for i, (sensor_data_id, data_object) in enumerate(data_objects):
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


def update_data(load_data):
    datatypes = 0
    # TODO (@gabvaztor) Create an object to store data object
    for i in range(datatypes):
        data.append([[], []])
        
    Reader(path=miranda_path, sensor_type=20, load_data=load_data).read_refresh_data()
    #Reader(path=miranda_path, sensor_type=1, load_data=load_data).read_refresh_data()
    """
    data = [[clamp_date_1, clamp_value_1], [clamp_date_2, clamp_value_2], [clamp_date_2, clamp_value_3],
            [presence_date, presence_value], [humidity_date, humidity_value],
            [luminosity_date, luminosity_value], [temperature_date, temperature_value]]
    """

    return data

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


def date_from_format(date, format):
    try:
        date = datetime.strptime(date, format)
    except:
        if len(date) > 19:
            fix_date = date[-19:]
            date = datetime.strptime(fix_date, format)
    return date

def update_data_object_information(end_date, last_filepath):
    for _, data_object in data_objects.items():
        sensor_id = data_object.information.sensor_id
        data_id = data_object.information.data_id
        measure = None
        data_type = None
        if sensor_id == 20:
            measure = Measures.WATIOS
            data_type = DataTypes.ELECTRIC_CLAMP
        elif sensor_id in [1, 2, 3, 4, 5, 6]:
            if data_id == 1:
                measure = Measures.WATIOS
                data_type = DataTypes.ELECTRIC_CLAMP
            elif data_id == 2:
                measure = Measures.TEMPERATURE
                data_type = DataTypes.TEMPERATURE
            elif data_id == 3:
                measure = Measures.HUMIDITY
                data_type = DataTypes.HUMIDITY
            elif data_id == 4:
                measure = Measures.LUMINOSITY
                data_type = DataTypes.LUMINOSITY
        data_object.information.set_info(measure=measure, datatype=data_type,
                                         file_path=last_filepath, end_date=end_date)

class Reader():

    def __init__(self, path, sensor_type, load_data):
        self.path = path
        self.sensor_type = sensor_type
        self.load_data = load_data

    def read_refresh_data(self, i=None):
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
        data_loaded_flag = self.load_historic_data(sorted_paths)

        # Info data
        first_date = None  # To see delta to show data
        end_date = None
        last_filepath = None
        total_files = len(sorted_paths)

        for file_count, file in enumerate(sorted_paths):
            pt("File " + str(file_count + 1) + " of " + str(total_files))
            # Open file
            file_data = open(file, "r").read()
            lines = file_data.split("\n")
            total_lines = len(lines)
            pir_count = 0
            pt("file", file)
            for line_count, line in enumerate(lines):
                if line_count % 5000 == 0:
                    gc.collect()
                printProgressBar(iteration=line_count, total=total_lines, prefix='File progress:', suffix='Complete',
                                 length=50)
                if file_count == total_files - 1 and line_count == 0 and total_files > 1:
                    try:
                        last_filename = sorted_paths[file_count - 1]
                        filename = os.path.splitext(last_filename)[0]
                        fdsf
                        self.save_data_checkpoint(filename=filename)
                    except:
                        pt("Not saved")
                if len(line) > 1:
                    date_id_value = line.split(sep=";")
                    date, ID, value = date_id_value[0], int(date_id_value[1]), date_id_value[2]
                    sensor_data_id = str(self.sensor_type) + "_" + date_id_value[1]
                    date = date_from_format(date=date, format="%Y-%m-%d %H:%M:%S")
                    if not sensor_data_id in data_objects:  # It means there is a new sensor data type.
                        information = InfoDataObject(sensor_id=self.sensor_type, start_date=date, data_id=ID)
                        data_objects[sensor_data_id] = DataObject(information=information)

                    data_object = data_objects[sensor_data_id]

                    if file_count == 0 and line_count == 0:  # TODO (@gabvaztor) If not loaded before
                        first_date = date
                    delta_minutes = (date - first_date).total_seconds()/60
                    if delta_minutes > 10. or data_loaded_flag:
                        value = float(value)
                        if self.sensor_type == 20:  # ElectricClamp
                            if ID == 1:
                                if len(date_id_value) > 3:
                                    try:
                                        clamp_1_value = value
                                        clamp_2_value = float(date_id_value[3])
                                        clamp_3_value = float(date_id_value[4])
                                        total_value = value + clamp_2_value + clamp_3_value
                                        multiple_y_values = [clamp_1_value, clamp_2_value, clamp_3_value, total_value]
                                        data_object.add(x=date, multiple_y_values=multiple_y_values)
                                        # TODO (@gabvaztor)  FINISH THIS
                                        """
                                        clamp_1.add(x=date, y=value)
                                        clamp_2.add(x=date, y=clamp_2_value)
                                        clamp_3.add(x=date, y=clamp_3_value)
                                        clamp_total.add(x=date, y=total_value)

                                        data[0][0].append(date)
                                        data[0][1].append(value)
                                        data[1][0].append(date)
                                        data[1][1].append(clamp_2)
                                        data[2][0].append(date)
                                        data[2][1].append(clamp_3)
                                        data[3][0].append(date)
                                        data[3][1].append(total_value)
                                        """
                                    except Exception as e:
                                        pt("ERROR", e)

                        elif self.sensor_type == 1:  # Luminosity, presence, humidity and temperature
                            if ID == 1:  # Presence
                                presence = Presence(sensor=None, state=value)
                                data[7][0].append(date)
                                data[7][1].append(value)
                                pir_count = 0
                            elif ID == 2:  # Temperature
                                temperature = Temperature(sensor=None, celsius=value)
                                data[4][0].append(date)
                                data[4][1].append(value)
                                pir_count += 1
                                if pir_count > 2:
                                    data[7][0].append(date)
                                    data[7][1].append(0.)
                            elif ID == 3:  # Humidity
                                humidity = Humidity(sensor=None, relative=value)
                                data[5][0].append(date)
                                data[5][1].append(value)
                                pir_count += 1
                            elif ID == 4:  # Luminosity
                                luminosity = Luminosity(sensor=None, lux=value)
                                data[6][0].append(date)
                                data[6][1].append(value)
                    # If it is last line:
                    if line_count == total_lines - 1 and file_count == total_files - 2 and total_files > 1:
                        end_date = date
                        last_filepath = file
        update_data_object_information(end_date=end_date, last_filepath=last_filepath)

    def load_historic_data(self, sorted_paths):
        """
        First, we get all saved files if exists. After that, we remove the appropiates files in "sorted_paths" to not
        read that file (and not load its data).

        Args:
            sorted_paths: Sorted paths with all paths with interested data.

        """
        global data
        data_loaded = False
        if load_data:
            for fullpath, root, name in get_files_from_path(paths=checkpoint_path, ends_in=".npy"):
                pt("Loading data...")
                info = name.split("_")
                sensor_id = info[0]
                date = datetime.strptime(info[1], "%Y%m%d")
                for index, sorted_path in enumerate(sorted_paths.copy()):
                    info_sorted = os.path.splitext(os.path.basename(sorted_path))[0].split("_")
                    sensor_id_sorted = info_sorted[0]
                    if sensor_id_sorted == sensor_id:
                        date_sorted = datetime.strptime(info_sorted[1], "%Y%m%d")
                        if date_sorted <= date:
                            sorted_paths.remove(sorted_path)
                temp_data = list(np.load(fullpath))
                data = temp_data
                data_loaded = True
                pt("Data loaded")
        return data_loaded

    def save_data_checkpoint(self, filename):
        """
        Args:
            filename: Filename to save
        """
        pt("Saving data...")
        save_fullpath = checkpoint_path + filename + ".npy"
        create_directory_from_fullpath(save_fullpath)
        np.save(file=save_fullpath, arr=np.asarray(data))
        pt("Data saved")

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
load_data = False
update_data(load_data=load_data)
datatypes = calculate_datatypes() # Number of different data
calculate_deltas_from_data()
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
    for doid, data_object in data_objects.items():
        for i in range(data_object.number_datatypes):
            pt("Creating graph " + str(i + 1) + " of " + str(len(data_object.number_datatypes)) + "...")
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
            add_to_graph(title=data_object.information.datatype[i])
            #figures.append(fig)
            #graphs.append(subplot)
            subplot.clear()
            lines = subplot.plot(data_object.x[i], data_object.y[i], "-")
            l1 = lines
            plt.setp(lines, linestyle='-')  # set both to dashed
            plt.setp(l1, linewidth=.08, color='b')  # line1 is thick and red
            formatter = mdates.DateFormatter("%m/%d %H:%M:%S")
            ylabel = ylabels[i] + " | Max Δ = " + "{0:.2f}".format(delta_data[i])
            subplot.set_ylabel(ylabel)
            subplot.xaxis.set_major_formatter(formatter)
            fig.autofmt_xdate()
            # Save graph
            save_path_graph = save_path + "Graphs\\" + actual_time + "\\" + titles[i].replace(" ", "_") + ".png"
            create_directory_from_fullpath(fullpath=save_path_graph)
            fig.savefig(fname=save_path_graph, dpi=1500)
            gc.collect()
    #animated_clamp = animation.FuncAnimation(figures[0], clamp_function, interval=1000000000)
    #animated_global_sensor = animation.FuncAnimation(figures[1], global_sensor_function, interval=10000000)



    """
    for i, graph in enumerate(graphs):
        pt("Creating graph " + str(i + 1) + " of " + str(len(graphs)) + "...")
        graph.clear()
        if data[i][0]:
            lines = graph.plot(data[i][0], data[i][1], "-")
            l1 = lines
            plt.setp(lines, linestyle='-')  # set both to dashed
            plt.setp(l1, linewidth=.08, color='b')  # line1 is thick and red
            formatter = mdates.DateFormatter("%m/%d %H:%M:%S")
            ylabel = ylabels[i] + " | Max Δ = " + "{0:.2f}".format(delta_data[i])
            graph.set_ylabel(ylabel)
            graph.xaxis.set_major_formatter(formatter)
            figures[i].autofmt_xdate()
            # Save graph
            save_path_graph = save_path + "Graphs\\" + actual_time + "\\" + titles[i].replace(" ", "_") + ".png"
            create_directory_from_fullpath(fullpath=save_path_graph)
            figures[i].savefig(fname=save_path_graph, dpi=1500)
            gc.collect()
    """
