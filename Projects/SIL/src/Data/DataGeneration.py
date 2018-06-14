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

def generate_data(data, column_name, frequency, function, seconds=None, end_date=None):
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
    now = datetime.now()
    print("###################################")
    print("Generating data")
    if end_date:
        datetimes = [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in
                     datetime_range(start=datetime(2018, 1, 1, 0),
                                    end=end_date,
                                    delta_minutes=timedelta(minutes=minutes))]
    else:
        datetimes = [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in
                     datetime_range(start=datetime(2018, 1, 1, 0),
                                    end=datetime(2018, 1, 7, 0),
                                    delta_minutes=timedelta(minutes=minutes))]
    print("Time to generate data: ", (datetime.now() - now).total_seconds())
    labels = [column_name, "Datatime"]
    total_data = len(datetimes)
    if not data:
        data = ['%.5f' % x for x in list(np.random.random_sample(size=total_data,))]
    else:
        data = data
    table = [data, datetimes]
    now = datetime.now()
    dataframe = pd.DataFrame(table)
    dataframe = dataframe.transpose()
    dataframe.columns = labels
    print("Dataframe generation and operations", (datetime.now() - now).total_seconds())
    return  dataframe

def save_to_files(filename, dataframe):

    # CSV
    now = datetime.now()
    dataframe.to_csv(path_or_buf="../../data/raw/" + filename + ".csv", sep=',')
    print("To CSV", (datetime.now() - now).total_seconds())
    # TXT
    now = datetime.now()
    np.savetxt(fname="../../data/raw/" + filename + ".txt", X=dataframe.values, fmt='%s')
    print("To TXT", (datetime.now() - now).total_seconds())
    # NPY
    now = datetime.now()
    np.save(file="../../data/raw/" + filename + ".npy", arr=dataframe.values)
    print("To NPY", (datetime.now() - now).total_seconds())
    # JSON
    now = datetime.now()
    dataframe.to_json(path_or_buf="../../data/raw/" + filename + ".json")
    print("To JSON", (datetime.now() - now).total_seconds())
    # pickle
    now = datetime.now()
    dataframe.to_pickle(path="../../data/raw/" + filename + ".pkl")
    print("To pickle", (datetime.now() - now).total_seconds())
    # msgpack
    now = datetime.now()
    dataframe.to_msgpack(path_or_buf="../../data/raw/" + filename + ".msg")
    # to_feather --> Need to install library feather
    #dataframe.to_feather(fname="../../data/raw/" + filename + ".feather")
    # HDF --> Need to install library tables : pip3 install tables
    #dataframe.to_hdf("../../data/raw/" + filename + ".hdf", 'test', mode='w')
    print("To to_msgpack", (datetime.now() - now).total_seconds())
    # cpickle
    now = datetime.now()
    import _pickle as pty
    with open("../../data/raw/" + filename + ".cpickle", 'wb') as pickle_file:
        pty.dump(obj=dataframe, file=pickle_file)
    print("To _pickle", (datetime.now() - now).total_seconds())
    # gzip --> best with memory requeriments and compression and decompression time
    import gzip
    now = datetime.now()
    with gzip.open("../../data/raw/" + filename + ".gz", 'wb') as f:
        f.write(dataframe.values)
    print("gzip time to compress: ", (datetime.now() - now).total_seconds())
    # lzma compreesor  --> compress with lower size
    now = datetime.now()
    import lzma
    now = datetime.now()
    with lzma.open("../../data/raw/" + filename + ".xz", 'wb') as compress_file:
        compress_file.write(dataframe.values)
    print("lzma time to compress: ", (datetime.now() - now).total_seconds())

# f = 3600s
dt = generate_data(data=None, column_name="Float", frequency=None, function=None, seconds=3600)
print(dt.shape)
# Save to files
filename = "1month_3600s"
save_to_files(filename=filename, dataframe=dt)
print("3600s, 1month")

# f = 60s
dt = generate_data(data=None, column_name="Float", frequency=None, function=None, seconds=60)
print(dt.shape)
# Save to files
filename = "1month_60s"
save_to_files(filename=filename, dataframe=dt)
print("60s, 1month")

# f = 1s
dt = generate_data(data=None, column_name="Float", frequency=None, function=None, seconds=1,
                   end_date=datetime(2018, 1, 7, 0))
print(dt.shape)
# Save to files
filename = "7days_1s"
save_to_files(filename=filename, dataframe=dt)
print("1s, 7days")