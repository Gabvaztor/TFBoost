class Humidity():
    """
    Humidity class
    """
    def __init__(self, sensor, relative, absolute=None, specific=None ):
        self.sensor = sensor
        self.absolute = absolute
        self.specific = specific
        self.relative = relative