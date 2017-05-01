"""
Author: @gabvaztor
StartDate: 04/03/2017

With this class you can pre-process your data. For example,
you can add statistical information or order your data.

Style: "Google Python Style Guide" 
https://google.github.io/styleguide/pyguide.html
"""

"""
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
"""

# --------------------------------------------------------------------------
"""Reader class
"""
from TFBoost.TFReader import Reader
# --------------------------------------------------------------------------

class DataMining(Reader):
    # TODO DOCS
    """DataMining Class

    With this class you can add a lot of information to your data. For example,
    you can add statistical information, order your data, reducing noise and others.

    To access DataMining class you have to create a Reader object before.

    Attributes:
    reader (obj:'Reader'): Reader Object. This attribute contains the sets to manipulate.
    chooses (:obj:'Strings List', optional): Contains how data will be manipulated
    """

    # TODO Defining Chooses

    reader = None
    chooses = []

    def __init__(self, reader, chooses = None):
        # TODO Define this
        """

        :param reader:
        :param chooses:
        """
        if reader:
            self.reader = reader
            if  chooses:
                self.chooses = chooses
                self.manipulate()

    def manipulate(self):
        """

        :return:
        """
        return self.reader

if __name__ == '__main__':
    print ("Creating DataMining")