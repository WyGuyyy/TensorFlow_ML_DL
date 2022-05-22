from keras import models
from keras import layers
from keras import regularizers

# Original network
# model = models.Sequential()
# model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

# Smaller network (wtih lower capacity)
# This smaller network ensures that
# *******************************************************************
# model = models.Sequential()
# model.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(4, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
# *******************************************************************

# Larger network (with larger capacity)
# model = models.Sequential()
# model.add(layers.Dense(512, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

# Original network with regularizers
# model = models.Sequential()
# model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
#                       activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
#                      activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

# Original network with dropout
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

# Takeaways/Follow ups
# 1.) Study up on l1 and l4 regularization again
# 2.) Study up on some of the activiation and loss functions

