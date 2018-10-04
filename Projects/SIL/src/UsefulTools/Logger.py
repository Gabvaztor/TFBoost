import logging
from datetime import datetime
from UsefulTools.UtilsFunctions import create_file_from_fullpath
import traceback as traceback_

def logger(warning, traceback=None):
    """
    Creates a logging object and returns it
    """
    try:
        logger = logging.getLogger("debug_logger")
        logger.setLevel(logging.DEBUG)

        actual_datetime = datetime.now().strftime("%Y-%m-%d")
        file = r"..//..//"
        file_name = actual_datetime + "_info_logger.log"
        full_file = file + file_name
        create_file_from_fullpath(full_file)
        # create the logging file handler
        fh = logging.FileHandler(full_file)

        fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(fmt)
        fh.setFormatter(formatter)

        # add handler to logger object
        logger.addHandler(fh)
        logger.debug(str(warning))

        if traceback:
            string_traceback = repr(traceback.extract_stack())
            logger.debug(string_traceback)
    except:
        traceback_.print_exc()
        print("Currently it can not write to logger")