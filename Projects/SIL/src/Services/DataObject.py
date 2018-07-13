import numpy as np

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

    def data(self):
        return self.all_data[0]

    def x(self):
        if self.multiple_x:
            return self.multiple_x
        else:
            return self.data[0]

    def y(self):
        if self.multiple_y:
            return self.multiple_y
        else:
            return self.data[1]

    def info(self):
        return self.data[2]

    def number_datatypes(self):
        """
        This calculate and return the number of datatypes (different 'y' values) in the DataObject.

        Returns: number of datatypes
        """

        if self.y:
            return 1
        else:
            return len(self.multiple_y)

    def calculate_deltas(self):
        """
        Calcule the delta values between y values
        """
        if self.y:
            delta = max(self.y) - min(self.y)
            self.deltas.append(delta)
        elif self.multiple_y:
            for data_type in self.multiple_y:
                delta = max(data_type) - min(data_type)
                self.deltas.append(delta)
        else:
            self.deltas = []

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

    all_data = None
    datatypes = DataTypes()
    information = InfoDataObject()
    data = data()
    # TODO (@gabvaztor) Finish this
    x = x()
    y = y()
    info = info()
    multiple_x = []
    multiple_y = []
    deltas = calculate_deltas()  # Represents the deltas of data. If y, only one position. If multiple_y, multiple
    # positions for each y.
    number_datatypes = number_datatypes()