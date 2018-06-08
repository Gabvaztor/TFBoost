class Core():
    """
    Core class
    """
    def __init__(self, domotic_object, cpu_temperature, cpu_usage, ram_usage):
        self.domotic_object = domotic_object
        self.cpu_temperature = cpu_temperature
        self.cpu_usage = cpu_usage
        self.ram_usage = ram_usage