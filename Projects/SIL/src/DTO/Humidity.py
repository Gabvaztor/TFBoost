class Humidity():
    """
    Humidity class
    """
    def __init__(self, sensor, absolute, specific, relative):
        self.sensor = sensor
        self.absolute = absolute
        self.specific = specific
        self.relative = relative