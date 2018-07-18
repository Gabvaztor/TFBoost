import os
from UsefulTools.UtilsFunctions import get_temp_file_from_fullpath

class A():
    def __init__(self):
        self.data = self.data()
    def data(self):
        return self.all_data[0]
    all_data = [[1]]

a = A()
print(str(a.data))
a.data.append(51)
print(str(a.data))

