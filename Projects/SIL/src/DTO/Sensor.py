class Sensor():
    """
    Sensor class
    """
    def __init__(self, core, frequency_hz, frequency_hz_max, frequency_hz_min):
        self.core = core
        self.frequency_hz = frequency_hz
        self.frequency_hz_max = frequency_hz_max
        self.frequency_hz_min = frequency_hz_min