"""

Contains methods to generate data.

"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import pickle
# pip3 install -U feather-format

"""
df = pd.DataFrame(np.random.randn(10000, 100))

>>> %timeit df.to_pickle('test.p')
10 loops, best of 3: 22.4 ms per loop

>>> %timeit df.to_msgpack('test.msg')
10 loops, best of 3: 36.4 ms per loop

>>> %timeit pd.read_pickle('test.p')
100 loops, best of 3: 10.5 ms per loop

>>> %timeit pd.read_msgpack('test.msg')
10 loops, best of 3: 24.6 ms per loop
"""

def datetime_range(start, end, delta_minutes):
    current = start
    while current < end:
        yield current
        current += delta_minutes

def frequency_to_seconds(frequency):
    return int(1/frequency)

def generate_data(data, column_name, frequency, function, seconds=None):
    # TODO (@gabvaztor)
    """
    Create a DataFrame that contains the creation of data of a data_type from a function.
    Args:
        data: data
        column_name: column name of dataframe
        frequency: hz of data
        function:
        seconds:

    Returns:
            Dataframe with data.
    """
    if not frequency:
        minutes = seconds / 60
    else:
        minutes = (frequency / 1000) / 60
    datetimes = [dt.strftime('%Y-%m-%d %H:%M:%S.%f') for dt in
                 datetime_range(start=datetime(2018, 1, 1, 0),
                                end=datetime(2018, 4, 1, 0),
                                delta_minutes=timedelta(minutes=minutes))]
    labels = [column_name, "Datatime"]
    total_data = len(datetimes)
    if not data:
        data = ['%.3f' % x for x in list(np.random.random_sample(size=total_data,))]
    else:
        data = data
    table = [data, datetimes]
    dataframe = pd.DataFrame(table)
    dataframe = dataframe.transpose()
    dataframe.columns = labels

    return  dataframe

def save_to_files(filename, dataframe):
    # CSV
    dataframe.to_csv(path_or_buf="../../data/raw/" + filename + ".csv", sep=',')
    # TXT
    np.savetxt(fname="../../data/raw/" + filename + ".txt", X=dataframe.values, fmt='%s')
    # NP
    np.save(file="../../data/raw/" + filename + ".npy", arr=dataframe.values)
    # JSON
    dataframe.to_json(path_or_buf="../../data/raw/" + filename + ".json")
    # pickle
    dataframe.to_pickle(path="../../data/raw/" + filename + ".pkl")
    # msgpack
    dataframe.to_msgpack(path_or_buf="../../data/raw/" + filename + ".msg")
    # to_feather --> Need to install library feather
    dataframe.to_feather(fname="../../data/raw/" + filename + ".feather")
    # HDF --> Need to install library tables : pip3 install tables
    dataframe.to_hdf("../../data/raw/" + filename + ".hdf", 'test', mode='w')
    # cpickle
    import _pickle as pty
    with open("../../data/raw/" + filename + ".cpickle", 'wb') as pickle_file:
        pty.dump(obj=dataframe, file=pickle_file)


"""
# f = 3600s
dt = generate_data(data=None, column_name="Floats", frequency=None, function=None, seconds=3600)
print(dt.shape)
# Save to files
filename = "1month_3600s"
save_to_files(filename=filename, dataframe=dt)

# f = 60s
dt = generate_data(data=None, column_name="Floats", frequency=None, function=None, seconds=60)
print(dt.shape)
# Save to files
filename = "1month_60s"
save_to_files(filename=filename, dataframe=dt)
"""
# f = 1s
dt = generate_data(data=None, column_name="Floats", frequency=None, function=None, seconds=1)
print(dt.shape)
# Save to files
filename = "1month_1s"
save_to_files(filename=filename, dataframe=dt)

