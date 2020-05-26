#%% Given code
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
N_train_samples = 600
N_validation_samples = 100
N_test_samples = 100
N_samples = N_train_samples + N_validation_samples + N_test_samples
noise_sig = 0.1
N_epochs = 150
batch_size = 8
learning_rate = 0.01

tf.random.set_seed(0)
np.random.seed(0)
x = np.linspace(0.0, 3.0, N_samples, dtype=np.float32)
y = np.expand_dims(np.sin(1.0+x*x) + noise_sig*np.random.randn(N_samples).astype(np.float32), axis=-1)
y_true = np.sin(1.0+x*x)

plt.plot(x, y)
plt.plot(x, y_true)
plt.legend(["Observation", "Ground truth"])
plt.show()
#%% Stacking idea for shuffling and partitioning
y_reshaped = y.reshape(1,800)
y_full = np.vstack((y_reshaped[0], y_true))
full_array  = np.vstack((y_full, x)).T
full_array.dtype
full_array.shape
np.random.shuffle(full_array)
full_array = full_array.T

#%% Shuffle the data witk sklearn
# from sklearn.utils import shuffle
# y_reshaped = y.reshape(1,800)
# y_shuffeled, x_shuffeled = shuffle(y_reshaped[0], x)
# len(y_shuffeled)
# len(x_shuffeled)
#%%  Create Tensorflow datasates
""" Shuffle and partition the data set accordingly. you can use the predefined constants "N_train_samples", "N_validation_samples" and "N_test_samples". Use the variable names that are already in the below code
to store the final shuffled and partitioned data. Hint: Shuffle the data and the labels in such a way that the pairing between an image and it's label is preserved."""
# Shuffle the data
y_reshaped = y.reshape(1,800)
y_full = np.vstack((y_reshaped[0], y_true))
full_array  = np.vstack((y_full, x)).T
# full_array.dtype
# full_array.shape
#%% Partition the data
np.random.shuffle(full_array)
""" full array[0] is y  , full array[1] is y_true , full array[2] is all shuffled the same way  , without the transpose operator an array of tuples is returned"""
full_array = full_array.T
""" when other packages are allowed another solution would be
from sklearn.utils import shuffle
y_reshaped = y.reshape(1,800)
y_shuffeled, x_shuffled = shuffle(y_reshaped[0], x)"""
#Partition the data
x_train = full_array[2, 0:N_train_samples]
y_train = full_array[0, 0:N_train_samples]
x_validation = full_array[2, N_train_samples:N_train_samples+ N_validation_samples]
y_validation = full_array[0, N_train_samples:N_train_samples+ N_validation_samples]
x_test = full_array[2, N_train_samples+ N_validation_samples:N_train_samples+ N_validation_samples +N_test_samples]
y_test = full_array[2, N_train_samples+ N_validation_samples:N_train_samples+ N_validation_samples +N_test_samples]
#%% DataSets
""" Create three tensorflow Dataset objects that can be used to feed the training test and validation data to a neural network. Hint: For the training data set use shuffling, batching with the size according to
the predefined constant "batch_size" and repeat the data set indefinetly. For the validation and test data sets no shuffling or batching is needed."""
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds=train_ds.shuffle(buffer_size = 600)
train_ds
validation_ds = tf.data.Dataset.from_tensor_slices((x_validation, y_validation))
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

#%% Do model
""" Implement a neural network with two hidden dense layers containing 10 neurons each. As an activation function use the tangens hyperbolicus (tf.nn.tanh()). Since we are not using Keras, we need to create and
manage all the variables that we need ourselves. The varaibles are created in the constructor of our model class. Since we want to be able to just call the class with some inputs in order to make a prediction,
we implement a __call__ method which computes the forward pass and returns the output of the network."""
# use tanh to activate
initializer = tf.initializers.glorot_uniform()
x.shape
# use tanh to activate
initializer([  10 ])
class MyModel(object):
    def __init__(self):
        # Create model variables
        self.W0 = tf.Variable(name = 'weight_0',trainable = True, dtype = tf.float32, initial_value = tf.Variable(tf.random.truncated_normal([1, 10])))
        self.b0 = tf.Variable(name = 'bias_0',trainable = True, dtype = tf.float32, initial_value = tf.Variable(tf.random.truncated_normal([ 10])))
        self.W1 = tf.Variable(name = 'weight_1',trainable = True, dtype = tf.float32, initial_value = tf.Variable(tf.random.truncated_normal([10, 10])))
        self.b1 = tf.Variable(name = 'bias_1',trainable = True, dtype = tf.float32, initial_value = tf.Variable(tf.random.truncated_normal([ 10])))
        self.W2 = tf.Variable(name = 'weight_2',trainable = True, dtype = tf.float32, initial_value = tf.Variable(tf.random.truncated_normal([10, 1])))
        self.b2 = tf.Variable(name = 'bias_2',trainable = True, dtype = tf.float32, initial_value = tf.Variable(tf.random.truncated_normal([ 1])))
        self.trainable_variables = [self.W0, self.b0, self.W1, self.b1, self.W2, self.b2]

    def __call__(self, inputs):
        # Compute forward pass
        output = tf.reshape(inputs, [-1, 1])
        output = tf.nn.tanh( tf.add(tf.matmul(output, self.W0), self.b0))
        output =tf.nn.tanh( tf.add(tf.matmul(output, self.W1), self.b1))
        output = tf.nn.tanh(tf.add(tf.matmul(output, self.W2), self.b2))
        return output
        return output
#%% initialize model
mdl = MyModel()
#%% Calculate y_pred
""" We want to plot a prediction on the complete data set with a model before training. For this make a prediction on the variable "x". """

y_pred = mdl.__call__(x)
plt.plot(x, y)
plt.plot(x, y_true)
plt.plot(x, y_pred.numpy())
plt.legend(["Observation", "Target", "Prediction"])
plt.show()
#%% Calculate one training step
""" For training we need to implement a function that executes one training step. Fill in the missing code pieces for this function."""

def train_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        y_pred = model.__call__(x)# Compute a prediction with "model" on the input "x"
        loss_val = tf.compat.v1.losses.mean_squared_error(y, y_pred) # Compute the Mean Squared Error (MSE) for the prediction "y_pred" and the targets "y"
    grads = tape.gradient(loss_val, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_val
#%% test train step
opt = tf.optimizers.RMSprop(learning_rate)
train_step(mdl, opt, x,y)
#%% Testarea
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds =train_ds.batch(batch_size=batch_size)
validation_ds = tf.data.Dataset.from_tensor_slices([(x_validation, y_validation)])
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
for x_v, y_v in validation_ds:
    y_pred = mdl.__call__(x_v)# Compute a prediction with "mdl" on the input "x_v"
    y_v = tf.reshape(y_v, [-1,1])
    validation_loss =tf.compat.v1.losses.mean_squared_error(y_v, y_pred) # Compute the Mean Squared Error (MSE) for the prediction "y_pred" and the targets "y_v"
tf.reduce_sum()


def train_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        y_pred = model.__call__(x)# Compute a prediction with "model" on the input "x"
        y = tf.reshape(y, [-1,1])
        loss_val = tf.compat.v1.losses.mean_squared_error(y, y_pred) # Compute the Mean Squared Error (MSE) for the prediction "y_pred" and the targets "y"
    grads = tape.gradient(loss_val, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_val
test = np.zeros(100)
tf.pow(tf.linalg.norm(test, ord= 'fro', axis = [1,2]),2)
    tf.nn.l2_loss()
