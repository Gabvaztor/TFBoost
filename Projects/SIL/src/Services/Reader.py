import time

from Projects.SIL.src.DTO.ElectricClamp import ElectricClamp
from Projects.SIL.src.DTO.Humidity import Humidity
from Projects.SIL.src.DTO.Luminosity import Luminosity
from Projects.SIL.src.DTO.Presence import Presence
from Projects.SIL.src.DTO.Temperature import Temperature
from UsefulTools.UtilsFunctions import pt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import matplotlib.dates as md
from datetime import datetime

datatypes = 5
figures = []
graphs = []

def add_graph(title=None, ylabel=None, ax=None):
    plt.title(title)
    plt.suptitle(title)

titles = ["Power disaggregation","Presence", "Temperature", "Humidity", "Luminosity"]
ylabels = ["Power (Kw/h)", "Cº", "Lux", "%", "True / False"]

for i in range(datatypes):

    fig = plt.figure()
    subplot = fig.add_subplot(1, 1, 1)
    add_graph(title=titles[i], ylabel=ylabels[i])
    figures.append(fig)
    graphs.append(subplot)
"""
presence_graph = fig.add_subplot(1,1,1)
luminosity_graph = fig.add_subplot(1,1,1)
humidity_graph = fig.add_subplot(1,1,1)
temperature_graph = fig.add_subplot(1,1,1)
"""
"""
def read_data(i=None):
    # Open file
    lines = open("\\\\192.168.1.62\\miranda\\10_20180702.dat", "r").read().split("\n")
    # lines = read_changing_data(file)
    xs = []
    ys = []
    for line in lines:
        if len(line) > 1:
            date_id_value = line.split(sep=";")
            pt("data", date_id_value)
            date, ID, value = date_id_value[0], date_id_value[1], date_id_value[2]
            xs.append(date)
            ys.append(value)
    ax1.clear()
    ax1.plot(xs, ys)
"""
class Reader():

    def __init__(self, filepath, sensor_type):
        self.filepath = filepath
        self.sensor_type = sensor_type

    def read_refresh_data(self, i=None):

        pt("Refresing data...")

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

        # Open file
        #with open(self.filepath, "r") as file:
        #lines = read_changing_data(file)
        data = open(self.filepath, "r").read()
        lines = data.split("\n")
        for i, line in enumerate(lines):
            if len(line) > 1:
                date_id_value = line.split(sep=";")
                pt("data", date_id_value)
                date, ID, value = date_id_value[0], int(date_id_value[1]), date_id_value[2]
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
                date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                if self.sensor_type == 0:  # ElectricClamp
                    clamp = ElectricClamp(sensor=None, power=value, date=date)
                    clamp_date.append(date)
                    clamp_value.append(value)
                elif self.sensor_type == 1:  # Luminosity, presence, humidity and temperature
                    if ID == 1:  # Presence
                        presence = Presence(sensor=None, state=value)
                        presence_date.append(date)
                        presence_value.append(value)
                    elif ID == 2:  # Temperature
                        temperature = Temperature(sensor=None, celsius=value)
                        temperature_date.append(date)
                        temperature_value.append(value)
                    elif ID == 3:  # Humidity
                        humidity = Humidity(sensor=None, relative=value)
                        humidity_date.append(date)
                        humidity_value.append(value)
                    elif ID == 4:  # Luminosity
                        luminosity = Luminosity(sensor=None, lux=value)
                        luminosity_date.append(date)
                        luminosity_value.append(value)
        if clamp_date:
            graphs[0].clear()
            graphs[0].plot(clamp_date, clamp_value)
        if presence_date:
            graphs[1].clear()
            graphs[1].plot(presence_date, presence_value)
        if temperature_date:
            graphs[2].clear()
            graphs[2].plot(temperature_date, temperature_value)
        if humidity_date:
            graphs[3].clear()
            graphs[3].plot(humidity_date, humidity_value)
        if luminosity_date:
            graphs[4].clear()
            graphs[4].plot(luminosity_date, luminosity_value)
        plt.gcf().autofmt_xdate()

clamp_filepath="\\\\192.168.1.62\\miranda\\10_20180702.dat"
clamp_filepath="E:\\SmartIotLabs\\DATA\\nEW\\10_20180702.dat"
global_sensor_filepath="\\\\192.168.1.62\\miranda\\1_20180702.dat"
global_sensor_filepath="E:\\SmartIotLabs\\DATA\\nEW\\1_20180702.dat"
pt("clamp_filepath", clamp_filepath)
pt("global_sensor_filepath", global_sensor_filepath)

clamp_function = Reader(filepath=clamp_filepath, sensor_type=0).read_refresh_data
global_sensor_function = Reader(filepath=global_sensor_filepath, sensor_type=1).read_refresh_data

animated_clamp = animation.FuncAnimation(figures[0], clamp_function, interval=1000000000)
animated_global_sensor = animation.FuncAnimation(figures[1], global_sensor_function, interval=10000000)
plt.show()