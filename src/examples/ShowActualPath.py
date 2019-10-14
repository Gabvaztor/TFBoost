import os
def show_actual_path():
    print("Actual Path: \n", os.path.dirname(os.path.abspath(__file__)))
    print("Actual Path: \n", os.getcwd())
show_actual_path()