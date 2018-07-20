import numpy as np
from UsefulTools.UtilsFunctions import *
from sys import getsizeof

class Sensor():
    """
    Represent Sensors information.
    """
    type_0 = [20]
    type_1 = [1,2,3,4,5,6]

    @staticmethod
    def clamp_types():
        return [20]

    @staticmethod
    def global_sensor():
        return [1,2,3,4,5,6]

    @staticmethod
    def sensors_ids():
        """
        sensors_ids must represent all different ids in all types of
                        # sensors
        Returns: All kinds of sensors ids.

        """
        return Sensor.clamp_types() + Sensor.global_sensor()

class DataTypes():

    TEMPERATURE = "TEMPERATURE"
    HUMIDITY = "HUMIDITY"
    ELECTRIC_CLAMP = "ELECTRIC_CLAMP"
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

        if datatype:
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

    def set_info(self, datatype=None, measure=None, sensor_id=None, data_id=None, start_date=None, end_date=None,
                 file_path=None, client_id=None):

        if datatype:
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

    all_data = None
    datatypes = DataTypes()
    information = InfoDataObject()
    multiple_x = []
    multiple_y = []

    def __init__(self, information=None):
        self.set_data(self.create_data())

        if information:
            self.information = information

    def set_data(self, data):
        self.all_data = data

    def get_all_data(self):
        return self.all_data

    @staticmethod
    def create_data(first=None, second=None, third=None):
        if not first:
            first = []
        if not second:
            second = []
        if not third:
            third = InfoDataObject().array()

        return np.array([first, second, third])

    @property
    def data(self):
        return self.all_data

    @property
    def x(self):
        if self.multiple_x:
            return self.multiple_x
        else:
            return self.data[0]

    @property
    def y(self):
        if self.multiple_y:
            return self.multiple_y
        else:
            return self.data[1]

    @property
    def info(self):
        return self.data[2]

    @property
    def deltas_max_min(self):
        """
        Calcule deltas values between 'y' values
        """
        deltas = []
        max_value = 0.
        min_value = 0.
        if self.y:
            if type(self.y[0]) != type([]):
                max_value, min_value = max(self.y), min(self.y)
                deltas.append(max_value - min_value)
            else:
                for data_type in self.y:
                    max_value, min_value = max(data_type), min(data_type)
                    delta = max_value - min_value
                    deltas.append(delta)
        return [deltas, max_value, min_value]

    @property
    def number_datatypes(self):
        """
        This calculate and return the number of datatypes (different 'y' values) in the DataObject.

        Returns: number of datatypes
        """
        if self.multiple_y:
            return len(self.multiple_y)
        elif self.y:
            return 1
        else:
            return 0

    @property
    def title(self):
        """
        Returns:the title of this object
        """
        title_array = []
        if not is_none(self.information.datatype):
            title_array.append(str(self.information.datatype))
        if not is_none(self.information.end_date) and not is_none(self.information.start_date):
            try:
                difference = self.information.end_date - self.information.start_date
                title_array.append("(" + str(difference.days) + " days)")
            except:
                pt("Error in dates")
        if not is_none(self.information.sensor_id) and not is_none(self.information.data_id):
            title_array.append("[" + str(self.information.sensor_id) + "_" + str(self.information.data_id) + "]")
        return ''.join(elem for elem in title_array)

    @property
    def label(self, i=None):
        label_array = []
        separator = " | "
        if self.information.measure:
            label_array.append(self.information.measure + separator)
        if self.deltas_max_min[0]:
            if i:
                delta = self.deltas_max_min[0][i]
                label_array.append("Δ " + "{0:.2f}".format(delta) + separator)
        if self.deltas_max_min[1] > 0.:
            label_array.append("MAX {0:.2f}".format(self.deltas_max_min[1]) + separator)
        if self.deltas_max_min[1] > 0.:
            label_array.append("MIN {0:.2f}".format(self.deltas_max_min[2]) + separator)
        return ''.join(elem for elem in label_array)

    def pair_data(self, index):
        if self.multiple_y:
            if type(self.x[index]) != type(self.y[index]):  # This means that there is one value in x and more than one
                # in y
                return np.asarray(self.x), np.asarray(self.y[index])
            else:
                return np.asarray(self.x[index]), np.asarray(self.y[index])
        else:
            return np.asarray(self.x), np.asarray(self.y)

    def add(self, x=None, y=None, multiple_x_values=None, multiple_y_values=None):
        """
        Add a element or multiple element to x, y. Between x, y, multiple_x_values and multiple_y_values must sum 2
        not None elements.
        Args:
            x: x values
            y: y values
            multiple_x_values: multiple x values. Must be an array of elements
            multiple_y_data: multiple y values. Must be an array of elements
        """
        if self.__check_two_none_elements([x,y,multiple_x_values, multiple_y_values]):
            if x is not None:
                self.x.append(x)
            if y is not None:
                self.y.append(y)
            if multiple_x_values is not None:
                if len(multiple_x_values) != len(self.multiple_x):
                    for _ in range(len(multiple_x_values)):
                        self.multiple_x.append([])
                for i, value in enumerate(multiple_x_values):
                    self.multiple_x[i].append(value)
            if multiple_y_values is not None:
                if len(multiple_y_values) != len(self.multiple_y):
                    for _ in range(len(multiple_y_values)):
                        self.multiple_y.append([])
                for i, value in enumerate(multiple_y_values):
                    self.multiple_y[i].append(value)
        else:
            pt("")

    def __check_two_none_elements(self, elements_list):
        count = 0
        for element in elements_list:
            if type(element) != type(None):
                count += 1
        if count > 2:
            pt("Can not be more than 2 element to add.")
        if count == 2:
            return True

    def total_bytes(self):
        total_bytes = 0
        for attribute in dir(self):
            total_bytes += getsizeof(eval(attribute))
            pt("attr", attribute)
        pt("total_bytes", total_bytes)
        pt("all dir)", dir())
        return total_bytes