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

datatypes = 8
figures = []
graphs = []

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

def calculate_deltas_from_data(data):
    for i, data_type in enumerate(data):
        if data_type[1]:
            value_list = data_type[1]
            delta_data.append(max(value_list) - min(value_list))
        else:
            delta_data.append(0)

def update_data(load_data):
    # TODO (@gabvaztor) Create an object to store data object
    for i in range(datatypes):
        data.append([[], []])
        
    Reader(path=miranda_path, sensor_type=20, load_data=load_data).read_refresh_data()
    Reader(path=miranda_path, sensor_type=1, load_data=load_data).read_refresh_data()
    """
    data = [[clamp_date_1, clamp_value_1], [clamp_date_2, clamp_value_2], [clamp_date_2, clamp_value_3],
            [presence_date, presence_value], [humidity_date, humidity_value],
            [luminosity_date, luminosity_value], [temperature_date, temperature_value]]
    """
    calculate_deltas_from_data(data)

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
        for file_count, file in enumerate(sorted_paths):
            pt("File " + str(file_count + 1) + " of " + str(len(sorted_paths)))
            # Open file
            file_data = open(file, "r").read()
            lines = file_data.split("\n")
            total_lines = len(lines)
            pir_count = 0
            pt("file", file)
            for line_count, line in enumerate(lines):
                if line_count % 5000 == 0:
                    gc.collect()
                if file_count == len(sorted_paths) - 1 and line_count == 0 and len(sorted_paths) > 1:
                    try:
                        last_filename = sorted_paths[file_count - 1]
                        self.save_data_checkpoint(os.path.splitext(last_filename)[0])
                    except:
                        pt("Not saved")
                printProgressBar(iteration=line_count, total=total_lines, prefix='File progress:', suffix='Complete',
                                 length=50)
                if len(line) > 1:
                    date_id_value = line.split(sep=";")
                    date, ID, value = date_id_value[0], int(date_id_value[1]), date_id_value[2]
                    sensor_data_id = str(self.sensor_type) + "_" + date_id_value[1]
                    if not sensor_data_id in data_objects:
                        data_objects[sensor_data_id] = DataObject()
                    data_object = data_objects[sensor_data_id]
                    try:
                        date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                    except:
                        if len(date) > 19:
                            fix_date = date[-19:]
                            date = datetime.strptime(fix_date , "%Y-%m-%d %H:%M:%S")
                    if file_count == 0 and line_count == 0:
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
            # End for
            # Data objects
            info_clamp_1 = InfoDataObject(measure=Measures.WATIOS,
                                          datatype=DataTypes.ELECTRIC_CLAMP_1,
                                          sensor_id=self.sensor_type,
                                          start_date=first_date,
                                          end_date=end_date, file_path=file,
                                          client_id="X").array()
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

    for i, type_data in enumerate(data):
        if i != 0 and i != 1:
            type_data = np.asarray(type_data)
            #histogram(type_data[1], iteration=i, save=save)
            #fourier(type_data[1], iteration=i, save=save)
            log_transormation(type_data, iteration=i, save=save)
    asfdaf
    return data

matplotlib = True
plotlylib = False
load_data = True
update_data(load_data=load_data)
#data = statistical_process(data=data, algorithm=1)

if plotlylib:
    for d in data:

        trace = dict(x=d[0], y=d[1])
        data_ = [trace]
        layout = dict(
            title='Time series with range slider and selectors'
        )
        fig = dict(data=data_, layout=layout)
        plotly.plot(fig)

if matplotlib:  # Matplot lib

    for i in range(datatypes):
        fig = plt.figure(figsize=(7, 7), dpi=90, facecolor='w', edgecolor='k')
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
        add_to_graph(title=titles[i])
        figures.append(fig)
        graphs.append(subplot)
    #animated_clamp = animation.FuncAnimation(figures[0], clamp_function, interval=1000000000)
    #animated_global_sensor = animation.FuncAnimation(figures[1], global_sensor_function, interval=10000000)

    actual_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    pt("Creating and saving graphs...")
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

class DataTypes():

    TEMPERATURE = "TEMPERATURE"
    HUMIDITY = "HUMIDITY"
    ELECTRIC_CLAMP_TOTAL = "ELECTRIC_CLAMP_TOTAL"
    ELECTRIC_CLAMP_1 = "ELECTRIC_CLAMP_1"
    ELECTRIC_CLAMP_2 = "ELECTRIC_CLAMP_2"
    ELECTRIC_CLAMP_3 = "ELECTRIC_CLAMP_3"
    LUMINOSITY = "LUMINOSITY"
    PRESENCE = "PRESENCE"

class Measures():

    CELSIUS = "ºC"
    PERCENT = "%"
    WATIOS = "W"
    LUX = "LUX"
    BOOLEAN = "TRUE/FALSE"

    TEMPERATURE = CELSIUS
    HUMIDITY = PERCENT
    LUMINOSITY = LUX
    PRESENCE = BOOLEAN
    CLAMP = WATIOS

class InfoDataObject():
    """
    Represent the data information. It will be in the third position of an DataObject
    """

    datatype = None
    measure = None
    sensor_id = None
    data_id = None
    start_date = None
    end_date = None
    file_path = None
    client_id = None

    def __init__(self, datatype=None, measure=None, sensor_id=None, data_id=None, start_date=None, end_date=None,
                 file_path=None, client_id=None):

        self.datatype = datatype
        self.measure = measure
        self.sensor_id = sensor_id
        self.data_id = data_id
        self.start_date = start_date
        self.end_date = end_date
        self.file_path = file_path
        self.client_id = client_id

    def set_info(self, datatype=None, measure=None, sensor_id=None, data_id=None, start_date=None, end_date=None,
                 file_path=None, client_id=None):

        if datatypes:
            self.datatype = datatype
        if measure:
            self.measure = measure
        if sensor_id:
            self.sensor_id = sensor_id
        if data_id:
            self.data_id = data_id
        if start_date:
            self.start_date = start_date
        if end_date:
            self.end_date = end_date
        if file_path:
            self.file_path = file_path
        if client_id:
            self.client_id = client_id

    def array(self):
        """Return an array of features"""
        return [self.datatype, self.measure, self.sensor_id, self.data_id, self.start_date, self.end_date,
                         self.file_path, self.client_id]

class DataObject():
    """
    'data' argument represents:
        - First array: date values (or x values)
        - Second array: sensor values (or y values)
        - Third array: DataObject information. The format is:
            ["datatype", "measure", "sensor_id", "data_id", "start_date", "end_date", "file_path", "client_id"]
            Example: ["TEMPERATURE", "Cº", "11", "1", "2018-02-02", "2018-02-03", "\\miranda\\sensor1.dat\\",
            "H5464356"]
            This is a "InfoDataObject"
    """

    datatypes = DataTypes()

    def __init__(self):
        self.set_data(self.create_data())

    def set_data(self, data):
        self.all_data = data

    @staticmethod
    def create_data(first=None, second=None, third=None):

        if not first:
            first = []
        if not second:
            second = []
        if not third:
            third = InfoDataObject().array()

        return np.array([first, second, third])

    def data(self):
        return self.all_data[0]

    def get_all_data(self):
        return self.all_data

    def x(self):
        return self.data[0]

    def y(self):
        return self.data[1]

    def info(self):
        return self.data[2]

    def add(self, x=None, y=None, multiple_x_values=None, multiple_y_values=None):
        """
        Add a element or multiple element to
        Args:
            x: x values
            y: y values
            multiple_x_values: multiple x values. Must be an array of elements
            multiple_y_data: multiple y values. Must be an array of elements

        """
        if x:
            self.x.append(x)
        if y:
            self.y.append(y)
        if multiple_x_values:
            if len(multiple_x_values) != len(self.multiple_x):
                for _ in range(len(multiple_x_values)):
                    self.multiple_x.append([])
            for i, value in enumerate(multiple_x_values):
                self.multiple_x[i].append(value)
        if multiple_y_values:
            if len(multiple_y_values) != len(self.multiple_y):
                for _ in range(len(multiple_y_values)):
                    self.multiple_y.append([])
            for i, value in enumerate(multiple_y_values):
                self.multiple_y[i].append(value)

    all_data = create_data()
    data = data()
    x = x()
    y = y()
    multiple_x = []
    multiple_y = []

