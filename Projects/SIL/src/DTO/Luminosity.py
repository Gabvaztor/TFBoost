class Luminosity():
    """
    Luminosity class
    """
    def __init__(self, sensor, lux, color_temperature=None):
        self.sensor = sensor
        self.lux = lux
        self.color_temperature = color_temperature