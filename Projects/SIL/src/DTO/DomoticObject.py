class DomoticObject():
    """
    DomoticObject class
    """
    def __init__(self, client, start_date, position, frequency_hz, frequency_hz_max, frequency_hz_min):
        self.client = client
        self.start_date = start_date
        self.position = position
        self.frequency_hz = frequency_hz
        self.frequency_hz_max = frequency_hz_max
        self.frequency_hz_min = frequency_hz_min