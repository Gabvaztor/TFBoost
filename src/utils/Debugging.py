string_separator = '-------------------------------------'

def pt(title=None, text=None):
    """
    Use the print function to print a title and an object coverted to string
    Args:
        title: head comment
        text: description comment
    """
    if text is None:
        text = title
        title = string_separator
    else:
        title += ':'
    print(str(title) + " \n " + str(text))