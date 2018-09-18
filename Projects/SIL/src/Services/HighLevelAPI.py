"""
This class contains all necessary attributes for represent "high" api values.
"""

class HighLevelObject():
    """
    Contains all highs variables that represents multiple ideas.
    """
    # TODO (@gabvaztor) Finish
    home_presence = 0  # Represents if someone is in home. 0 is False, 1 is True, 2 is unknown

    def __init__(self):
        # Constructor
        # TODO (@gabvaztor) Finish
        pass

    # TODO (@gabvaztor) Create sets functions
    pass

class HomePresenceObject():
    """
    Represents all events about Home Presence.
    """
    home_presence = 0
    sensor_id = None
    client_id = None
    multivariable_dict = None  #  Represents all variable to do operations
    metadata = {}  # Represents the date(key) and metadata_object(value). Date is key because we need to filter by time.

class HomePresenceSensorObject():
    """
    Represents all events about Home Presence.
    """
    home_presence = 0
    sensor_id = None
    client_id = None
    multivariable_dict = None  #  Represents all variable to do operations
    metadata = {}  # Represents the date(key) and metadata_object(value). Date is key because we need to filter by time.

class HomePresenceEvent():

    time = None
    after_delta = None
    before_delta = None
    sensor_id = None
    person_identifier = []  # Represents, if not none, the list of probabilities that means who person has caused
    # the event.

    def __init__(self, time, after_delta=None, previous_delta=None):
        """
        Args:
            time: Actual time (represents the event)
            after_delta: Delta time with next pir event
            previous_delta: Delta time with previous event
        """
        # TODO (@gabvaztor) Check types
        self.time = time
        self.after_delta = after_delta
        self.previous_delta = previous_delta
