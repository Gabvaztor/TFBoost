import time

from Projects.SIL.src.DTO.ElectricClamp import ElectricClamp
from Projects.SIL.src.DTO.Humidity import Humidity
from Projects.SIL.src.DTO.Luminosity import Luminosity
from Projects.SIL.src.DTO.Presence import Presence
from Projects.SIL.src.DTO.Temperature import Temperature
from UsefulTools.UtilsFunctions import pt, printProgressBar, get_files_from_path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import matplotlib.dates as mdates
from datetime import datetime
import plotly.plotly as plotly
import numpy as np

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
miranda_path = "\\\\192.168.1.62\\miranda\\"
# global_sensor_filepath="E:\\SmartIotLabs\\DATA\\nEW\\1_20180702.dat"
pt("clamp_filepath", clamp_filepath)
pt("global_sensor_filepath", global_sensor_filepath)

datatypes = 5
figures = []
graphs = []

# Info
titles = ["Power disaggregation", "Presence", "Humidity", "Luminosity","Temperature"]
ylabels = ["Power (Kw/h)", "True / False", "%" ,"Lux", "Cº"]

# Data
clamp_date = []
presence_date = []
humidity_date = []
luminosity_date = []
temperature_date = []

clamp_value = []
presence_value = []
humidity_value = []
luminosity_value = []
temperature_value = []

def add_graph(title=None, ylabel=None, ax=None):
    plt.title(title)
    plt.suptitle(title)

def update_data():


    # clamp_function = Reader(filepath=clamp_filepath, sensor_type=10).read_refresh_data
    Reader(path=miranda_path, sensor_type=1).read_refresh_data()
    data = [[clamp_date, clamp_value], [presence_date, presence_value], [humidity_date, humidity_value],
            [luminosity_date, luminosity_value], [temperature_date, temperature_value]]
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

    def __init__(self, path, sensor_type):
        self.path = path
        self.sensor_type = sensor_type

    def read_refresh_data(self, i=None):
        pt("Refresing data...")

        sorted_paths = []
        sorted_with_dates = []
        for fullpath, root, name in get_files_from_path(paths=self.path, ends_in=".dat"):
            if name.split("_")[0] == str(self.sensor_type):
                # Orders paths
                sorted_paths, sorted_with_dates = sort_paths_by_date_in_basename(sorted_paths,
                                                                                 sorted_with_dates,
                                                                                 fullpath,
                                                                                 name)
        first_date = None  # To see delta to show data
        for file_count, file in enumerate(sorted_paths):
            # Open file
            data = open(file, "r").read()
            lines = data.split("\n")
            total_lines = len(lines)
            pir_count = 0
            pt("file", file)
            for line_count, line in enumerate(lines):
                printProgressBar(iteration=line_count, total=total_lines, prefix='Progress:', suffix='Complete', length=50)
                if len(line) > 1:
                    date_id_value = line.split(sep=";")
                    #pt("data", date_id_value)
                    date, ID, value = date_id_value[0], int(date_id_value[1]), date_id_value[2]
                    date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                    if file_count == 0 and line_count == 0:
                        first_date = date
                    delta_minutes = (date - first_date).total_seconds()/60
                    if delta_minutes > 10.:
                        if "." in value:
                            value = float(value)
                        else:
                            value = int(value)
                        if self.sensor_type == 10:  # ElectricClamp
                            clamp = ElectricClamp(sensor=None, power=value, date=date)
                            clamp_date.append(date)
                            clamp_value.append(value)
                        elif self.sensor_type == 1:  # Luminosity, presence, humidity and temperature
                            if ID == 1:  # Presence
                                presence = Presence(sensor=None, state=value)
                                presence_date.append(date)
                                presence_value.append(value)
                                pir_count = 0
                            elif ID == 2:  # Temperature
                                temperature = Temperature(sensor=None, celsius=value)
                                temperature_date.append(date)
                                temperature_value.append(value)
                                pir_count += 1
                                if pir_count > 2:
                                    presence_date.append(date)
                                    presence_value.append(0)
                            elif ID == 3:  # Humidity
                                humidity = Humidity(sensor=None, relative=value)
                                humidity_date.append(date)
                                humidity_value.append(value)
                                pir_count += 1
                            elif ID == 4:  # Luminosity
                                luminosity = Luminosity(sensor=None, lux=value)
                                luminosity_date.append(date)
                                luminosity_value.append(value)


def statistical_process(data, algorithm):

    """

    Args:
        data: data
        algorithm:
            - 1: Histogram

    Returns: data processed

    """

    def histogram(vector):

        bins = np.delete(np.unique(vector), 0)
        pt("vector", vector)
        pt("bins", bins)
        ret = plt.hist(vector, bins=bins)
        plt.title("Histogram with 'auto' bins")
        plt.show()
        return ret

    for i, type_data in enumerate(data):  # Step = 2
        if i != 0:

            histogram(type_data[1])

    return data

matplotlib = False
plotlylib = True
data = update_data()
data = statistical_process(data=data, algorithm=1)

if plotlylib:
    trace = dict(x=luminosity_date, y=luminosity_value)
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
        add_graph(title=titles[i], ylabel=ylabels[i])
        figures.append(fig)
        graphs.append(subplot)
    #animated_clamp = animation.FuncAnimation(figures[0], clamp_function, interval=1000000000)
    #animated_global_sensor = animation.FuncAnimation(figures[1], global_sensor_function, interval=10000000)

    for i, graph in enumerate(graphs):
        if i != 0:
            graph.clear()
            graph.plot(data[i][0], data[i][1], "-")
            formatter = mdates.DateFormatter("%m/%d %H:%M:%S")
            graph.set_ylabel(ylabels[i])
            graph.xaxis.set_major_formatter(formatter)
            figures[i].autofmt_xdate()
    plt.show()
