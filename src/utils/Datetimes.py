from datetime import datetime

def date_from_format(date, format="%Y-%m-%d %H:%M:%S"):
    return date.strftime(format)