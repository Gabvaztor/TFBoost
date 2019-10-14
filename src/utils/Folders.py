import os

def write_string_to_pathfile(string, filepath):
    """
    Write a string to a path file
    :param string: string to write
    :param path: path where write
    """
    try:
        create_directory_from_fullpath(filepath)
        file = open(filepath, 'w+')
        file.write(str(string))
        return 1
    except:
        raise ValueError("Cannot write into folder")

def create_directory_from_fullpath(fullpath):
    """
    Create directory from a fullpath if it not exists.
    """
    directory = os.path.dirname(fullpath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory