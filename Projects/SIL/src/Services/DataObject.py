import numpy as np
from UsefulTools.UtilsFunctions import *
from sys import getsizeof
import traceback
import datetime

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
    unique_id = None

    def __init__(self, datatype=None, measure=None, sensor_id=None, data_id=None, start_date=None, end_date=None,
                 file_path=None, client_id=None, unique_id=None, information_array=None):

        if not is_none(information_array):
            for i, fact in enumerate(information_array):
                if i == 0:
                    self.datatype = information_array[0]
                elif i == 1:
                    self.measure = information_array[1]
                elif i == 2:
                    self.sensor_id = information_array[2]
                elif i == 3:
                    self.data_id = information_array[3]
                elif i == 4:
                    self.start_date = information_array[4]
                elif i == 5:
                    self.end_date = information_array[5]
                elif i == 6:
                    self.file_path = information_array[6]
                elif i == 7:
                    self.client_id = information_array[7]
                elif i == 8:
                    self.unique_id = information_array[8]
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
        if unique_id:
            self.unique_id = unique_id

    def set_info(self, datatype=None, measure=None, sensor_id=None, data_id=None, start_date=None, end_date=None,
                 file_path=None, client_id=None, unique_id=None, information_array=None):
        if not is_none(information_array):
            for i, fact in enumerate(information_array):
                if i == 0:
                    self.datatype = information_array[0]
                elif i == 1:
                    self.measure = information_array[1]
                elif i == 2:
                    self.sensor_id = information_array[2]
                elif i == 3:
                    self.data_id = information_array[3]
                elif i == 4:
                    self.start_date = information_array[4]
                elif i == 5:
                    self.end_date = information_array[5]
                elif i == 6:
                    self.file_path = information_array[6]
                elif i == 7:
                    self.client_id = information_array[7]
                elif i == 8:
                    self.unique_id = information_array[8]
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
        if unique_id:
            self.unique_id = unique_id

    def array(self):
        """Return an array of features"""
        return np.asarray(self.to_list())

    def to_list(self):
        return [self.datatype, self.measure, self.sensor_id, self.data_id, self.start_date, self.end_date,
                         self.file_path, self.client_id]

    def serializable(self, format="%Y-%m-%d %H:%M:%S"):
        information_list = self.to_list()
        for index, element in enumerate(information_list):
            if isinstance(element, datetime.date):
                information_list[index] = element.strftime(format)
        return information_list

class DataObject():
    """
    'data' argument represents:
        - First array: date values (or x values)
        - Second array: sensor values (or y values)
        - Third array: DataObject information. The format is:
            ["datatype", "measure", "sensor_id", "data_id", "start_date", "end_date", "file_path", "client_id",
            "unique_id"]
            Example: ["TEMPERATURE", "Cº", "11", "1", "2018-02-02", "2018-02-03", "\\miranda\\sensor1.dat\\",
            "H5464356", "11_1_20180202"]
            This is a "InfoDataObject"
    """

    all_data = None
    datatypes = DataTypes()
    information = InfoDataObject()

    def __init__(self, information=None, data_objects_list=None):
        self.multiple_x = []
        self.multiple_y = []
        self.set_data(self.create_data())
        if data_objects_list:
            for data_object in data_objects_list:
                self.join_data_object(data_object)

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
        try:
            if self.multiple_x:
                return self.multiple_x
            else:
                return self.data[0]
        except Exception:
            traceback.print_exc()

    @property
    def y(self):
        try:
            if self.multiple_y:
                return self.multiple_y
            else:
                return self.data[1]
        except Exception:
            traceback.print_exc()
    """
    @property
    def info(self):
        return self.information.array()
    """
    def is_valid(self):
        if len(self.x) > 0 and len(self.y) > 0:
            return True
        else:
            return False

    def start_save(self, fullpath):
        pt("Saving DataObject " + self.unique_doid + "...", "")
        try:
            create_directory_from_fullpath(fullpath)
            if len(self.data) == 3:
                self.data[2] = self.information.array()
                self.data[1] = np.asarray(self.y)
                self.data[0] = np.asarray(self.x)
            else:
                self.set_data(data=self.create_data(first=np.asarray(self.x),
                                             second=np.asarray(self.y),
                                             third=self.information.array()))
            np.save(file=fullpath, arr=np.asarray(self.data))
            pt("DataObject " + self.unique_doid + " saved without problems", "")
        except Exception:
            traceback.print_exc()
            pt("Problem saving data.")

    def start_load(self, fullpath):
        pt("Loading...")
        try:
            data = np.load(fullpath)
            self.set_data(data=data)
            self.information = InfoDataObject(information_array=data[2])
            if self.array_has_multiple_values(array=self.x):
                self.multiple_x = self.x
            if self.array_has_multiple_values(array=self.y):
                self.multiple_y = self.y
            return self
        except Exception:
            traceback.print_exc()
            pt("Problem loading data.")

    @property
    def unique_doid(self):
        if self.information.sensor_id and self.information.data_id:
            return str(self.information.sensor_id) + "_" + str(self.information.data_id)
        else:
            pt("Does not exist an unique ID")
            return "NOID_ERROR"

    @property
    def unique_doid_date(self):
        if is_none(self.information.unique_id):
            string_date = self.information.start_date.strftime("%Y%m%d")
            return self.unique_doid + "_" + string_date
        else:
            return self.information.unique_id

    @property
    def deltas_max_min(self):
        """
        Calcule deltas values between 'y' values
        """
        deltas = []
        max_value = 0.
        min_value = 0.
        if len(self.y) > 0:
            if not self.array_has_multiple_values(array=self.y):
                max_value, min_value = max(self.y), min(self.y)
                deltas.append(max_value - min_value)
            else:
                for data_type in self.y:
                    max_value, min_value = max(data_type), min(data_type)
                    delta = max_value - min_value
                    deltas.append(delta)
        return [deltas, max_value, min_value]

    @staticmethod
    def array_has_multiple_values(array):
        to_return = False
        if len(array) > 0:
            if type(array[0]) == type([]):
                to_return = True
        return to_return

    @staticmethod
    def array_is_not_empty(array):
        if len(array) > 0:
            return True
        else:
            return False

    @property
    def number_datatypes(self):
        """
        This calculate and return the number of datatypes (different 'y' values) in the DataObject.

        Returns: number of datatypes
        """
        if self.multiple_y:
            return len(self.multiple_y)
        elif len(self.y):
            return 1
        else:
            return 0

    @property
    def title(self):
        """
        Returns: the title of this data_object
        """
        title_array = []
        if not is_none(self.information.datatype):
            title_array.append(str(self.information.datatype))
        if not is_none(self.information.end_date) and not is_none(self.information.start_date):
            try:
                difference = self.information.end_date - self.information.start_date
                title_array.append("(" + str(difference.days + 1) + " days)")
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
        if len(self.deltas_max_min[0]) > 0:
            if not is_none(i):
                delta = self.deltas_max_min[0][i]
            else:
                delta = self.deltas_max_min[0][0]
            label_array.append("Δ " + "{0:.2f}".format(delta) + separator)
        if self.deltas_max_min[1] > 0.:
            label_array.append("MAX {0:.2f}".format(self.deltas_max_min[1]) + separator)
        if self.deltas_max_min[1] > 0.:
            label_array.append("MIN {0:.2f}".format(self.deltas_max_min[2]) + separator)
        return ''.join(elem for elem in label_array)

    def pair_data(self, index=None):
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
            pt("Can not Add to DataObject")

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
        pt("all dir()", dir())
        return total_bytes

    def data_array(self):
        x, y = self.pair_data()
        return np.array([x, y, self.information.array()])

    def x_array(self, to_string=False):
        if to_string:
            return np.asarray(self.x_string_date())
        return np.asarray(self.x)

    def x_string_date(self):
        if self.multiple_x:
            multiple_x = []
            multiple_x.append(self.transform_to_string_date_array(data=data) for data in self.multiple_x)
            return multiple_x
        else:
            return self.transform_to_string_date_array(data=self.x)

    @staticmethod
    def transform_to_string_date_array(data, format="%Y-%m-%d %H:%M:%S", to_float=False):
        if to_float:
            return [DataObject.date_to_float(date.strftime(format)) for date in data]
        else:
            return [date.strftime(format) for date in data]

    @staticmethod
    def date_to_float(string_date):
        return float(string_date[-5:].replace(":", "."))

    def y_array(self):
        return np.asarray(self.y)

    def y_to_list(self):
        if any(isinstance(element, np.ndarray) for element in self.y):
            multiple_y = []
            for data in self.y:
                multiple_y.append(list(data))
            return multiple_y
        else:
            return list(self.y)

    def serialize(self):
        serialize_data_object = {}
        serialize_data_object["x"] = self.x_string_date()
        serialize_data_object["y"] = self.y_to_list()
        serialize_data_object["information"] = self.information.serializable()
        return serialize_data_object

    def serialize_to_bytes(self):
        import pickle
        pickle.dump(self.serialize(), open("save.p", "wb"))

    from numba import jit
    @jit
    def join_data_object(self, data_object):
        if self.array_has_multiple_values(data_object.x):
            for i, values in enumerate(data_object.x):
                self.multiple_x[i].extend(values)
        else:
            self.x.extend(list(data_object.x))
        if self.array_has_multiple_values(data_object.y):
            for i, values in enumerate(data_object.y):
                self.multiple_y[i].extend(values)
        else:
            self.x.extend(list(data_object.y))
        self.information.set_info(information_array=data_object.information.array(),
                                  end_date=data_object.information.end_date)

    def std(self, index=None):
        if index:
            return np.std(self.y[index])
        return np.std(self.y)

    def window_sliding_y(self, step_size=1, width=3):
        return np.hstack(self.y[i:1 + i + i - width:step_size] for i in range(0, width))

    def window_sliding_x(self, step_size=1, width=3):
        return np.hstack(self.x[i:1 + i + i - width:step_size] for i in range(0, width))

    def window_sliding_pair(self, step_size=1, width=3):
        return self.window_sliding_x(step_size=step_size, width=width), \
               self.window_sliding_y(step_size=step_size, width=width)

    def dataframe(self):
        import pandas
        columns_data = {"Dates": self.x_array(), self.information.datatype: self.y_array()}
        dataframe = pandas.DataFrame(data=columns_data)
        return dataframe

