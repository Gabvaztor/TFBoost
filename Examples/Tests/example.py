import os
from UsefulTools.UtilsFunctions import *

def check_file_exists_and_change_name(path, char="", index=None):
    """
    Check if file exists and, if exists, try to change the name to another with a higher 'index'. Example:
    --> filename = 'name(id).png'. If exists, then try to create a new filename with a new index.
    --> new filename = 'name(id)_1.png)'. This has '_' as 'char'. If not char, then go only the index.
    Args:
        path: filepath
        char: char to add
        index: actual index
    Returns: new path
    """
    if file_exists_in_path_or_create_path(path):
        name = os.path.splitext(path)[0]
        extension = os.path.splitext(path)[1]
        if index == 0 or is_none(index):
            index = 1
            chars_to_delete = None
        else:
            chars_to_delete = number_of_digits(index)
            index = int(name[-chars_to_delete:])
        if chars_to_delete:
            new_path = name[:-chars_to_delete] + char + str(index) + extension
        else:
            new_path = name + char + str(index) + extension
        pt("new_path", new_path)
        path = check_file_exists_and_change_name(path=new_path, char=char, index=index)
    return path

path = "E:\\Downloads\\system-design-primer-master.zip"
path = check_file_exists_and_change_name(path=path, char="_")
pt(path)