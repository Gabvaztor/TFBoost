import os
from UsefulTools.UtilsFunctions import *

def check_file_exists_and_change_name(path, i=None):
    if is_none(i):
        i = 1
    else:
        i += 1
    if file_exists_in_path_or_create_path(path):
        new_path = os.path.splitext(path)[0] + "_" + str(i) + os.path.splitext(path)[1]
        check_file_exists_and_change_name(new_path, i)
    else:
        return path

path = "E:\\Downloads\\system-design-primer-master.zip"
path = check_file_exists_and_change_name(path=path)