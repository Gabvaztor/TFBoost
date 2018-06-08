import random
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate data
random.seed(111)
rng = pd.date_range(start='2000', periods=209, freq='M')
ts = pd.Series(np.random.uniform(-10, 10, size=len(rng)), rng).cumsum()
ts.plot(c='b', title='Example Time Series')
plt.show()
print(ts.head(10))

# Convert data into array that can be broken up into training "batches"
# that we will feed into our RNN model.
ts_array = np.array(ts)
num_period = 20
f_horizon = 1

x_data = ts_array[:(len(ts_array))]

print("---------------------------")
