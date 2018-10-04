"""
Day enum
"""
from Projects.SIL.src.UsefulTools.Logger import logger
class Day():

    MONDAY = 1
    TUESDAY = 2
    WEDNESDAY = 3
    THURSDAY = 4
    FRIDAY = 5
    SATURDAY = 6
    SUNDAY = 7

    MONDAY_STRING = "MO"
    TUESDAY_STRING = "TU"
    WEDNESDAY_STRING = "WE"
    THURSDAY_STRING = "TH"
    FRIDAY_STRING = "FR"
    SATURDAY_STRING = "SA"
    SUNDAY_STRING = "SU"

    def get_string_day(self, number_day):
        string_day = None
        if number_day == 1:
            string_day = self.MONDAY_STRING
        elif number_day == 2:
            string_day = self.TUESDAY_STRING
        elif number_day == 3:
            string_day = self.WEDNESDAY_STRING
        elif number_day == 4:
            string_day = self.THURSDAY_STRING
        elif number_day == 5:
            string_day = self.FRIDAY_STRING
        elif number_day == 6:
            string_day = self.SATURDAY_STRING
        elif number_day == 7:
            string_day = self.SUNDAY_STRING
        else:
            logger("number_day must be in [1,7] but is --> " + str(number_day))
        return string_day

