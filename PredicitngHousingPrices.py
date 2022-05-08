from keras.datasets import boston_housing
from keras import layers
from keras import models
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# Normalize the data since the features for each data point may have varying numerical scales.
# We will use a pretty typical normalizing technique of calculating the mean of all the data points
# in the training set and subtracting that mean from each data point. Then we calculate the standard
# deviation of the training set (after subtracting the mean) and divide the entire data set by the
# calculated standard deviation.
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

# Defining a function to build a model every time we need to instantiate the same model
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    # As a reminder for myself, we are not adding an activation function here since this is a scalar regression model.
    # Adding an activation function would only constrain what values the output could take. A great example from the book
    # Deep Learning with Python by Francois Chollet is if we used the sigmoid function - that would constrain the output to
    # be 0 or 1, but we want to predict a continuous value of housing prices.
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
        return smoothed_points

# A good approach to validation when the data set size is relatively small is k-fold.
# We will split our data set into k partitions and each time rotate the training set and
# validation sets so that we can compute a worthwhile validation score on a reasonable
# amount of data points.
k = 4
num_val_samples = len(train_data) // k # NOTE: Double slash in Python is floor division - so will round down to nearest whole number
num_epochs = 80
all_scores = []
all_mae_histories = [] # mae = mean absolute error (absolute value of the difference between the predictions and the targets)

for i in range(k):
    print('processing fold #', i)

    # grabbing our validation partition from the current iteration first
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # Now grab our current training data for this iteration
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
        train_data[(i + 1) * num_val_samples:]],
        axis=0
    )

    # Now grab our current target data for this iteration
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
        train_targets[(i + 1) * num_val_samples:]],
        axis=0
    )

    model = build_model() # using our function we wrote earlier
    history = model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=1, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)
]

# Now we plot the model
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()



