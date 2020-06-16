import tensorflow as tf
import tensorflow.keras as k
import numpy as np
import matplotlib.pyplot as plt
import wget
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.random.set_seed(0)
np.random.seed(0)
batch_size = 128
epochs = 20
learning_rate = 0.001

(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()
x_train_mnist = np.expand_dims(x_train_mnist, axis=-1).astype(np.float32)
x_test_mnist = np.expand_dims(x_test_mnist, axis=-1).astype(np.float32)
# y_train_mnist=k.utils.to_categorical(y_train_mnist,num_classes=10)
# y_test_mnist=k.utils.to_categorical(y_test_mnist,num_classes=10)
y_train_mnist[0]
#%% Catastrophic forgetting explained
""" Shift the labels of the FashionMNIST data set to {10,...,19}. """

(x_train_fmnist, y_train_fmnist), (x_test_fmnist, y_test_fmnist) = tf.keras.datasets.fashion_mnist.load_data()

x_train_fmnist = np.expand_dims(x_train_fmnist, axis=-1).astype(np.float32)
y_train_fmnist = y_train_fmnist + 10# Shift labels of training data
x_test_fmnist = np.expand_dims(x_test_fmnist, axis=-1).astype(np.float32)
y_test_fmnist = y_test_fmnist +10# Shift labels of testing data
#%% Check if shifting was a success
plt_img = np.zeros((280, 280))
for i in range(10):
  for j in range(10):
    plt_img[i*28:(i+1)*28, j*28:(j+1)*28] = np.squeeze(x_train_fmnist[i*10+j])
plt.imshow(plt_img, cmap="gray")
plt.axis("off")
plt.show()

print("Labels")
print("MNIST: "+str(np.unique(y_test_mnist)))
print("FashionMNIST: "+str(np.unique(y_test_fmnist)))
""" Implement a custom callback that evaluates a model on two data sets at the end of an epoch and stores the results in a two separate lists. Hint: The Keras callback class always posses an assotiated model. You can
use it via the "self.model" attribute of the class. Hint: Evaluating the model will return a tuple containing two elements, i.e. (loss, acc).  """

class MyCallback(tf.keras.callbacks.Callback):
  # Get the two different data sets and create lists for storing results
  def __init__(self, x_0, y_0, x_1, y_1, batch_size):
    super(MyCallback, self).__init__()
    self.x_0 = x_0
    self.y_0 = y_0
    self.x_1 = x_1
    self.y_1 = y_1
    self.loss_0 =[]
    self.acc_0 = []
    self.loss_1 = []
    self.acc_1 =[]
    self.batch_size = batch_size

  def on_epoch_end(self, epoch, logs=None):
    # Evaluate the model on both data sets and store results
    print("\nStarting callback...")
    print("+----------------------+")
    print("| Data set 0           |")
    print("+----------------------+")
    metrics_0_loss, metrics_0_acc =self.model.evaluate(self.x_0, self.y_0, batch_size=self.batch_size) #Evaluate the model on "self.x_0" and "self.y_0" with "self.batch_size"
    self.loss_0.append(metrics_0_loss)
    self.acc_0.append(metrics_0_acc)
    # Append loss to the loss list "self.loss_0" and accuracy to the accuracy list "self.acc_0"
    print("+----------------------+")
    print("| Data set 1           |")
    print("+----------------------+")
    metrics_1_loss, metrics_1_acc =self.model.evaluate(self.x_1, self.y_1, batch_size=self.batch_size) #Evaluate the model on "self.x_1" and "self.y_1" with "self.batch_size"
    self.loss_1.append(metrics_1_loss)
    self.acc_1.append(metrics_1_acc) # Append loss to the loss list "self.loss_1" and accuracy to the accuracy list "self.acc_1"
    print("Callback completed...")

    """ Instantiate a MyExtendedModel object, a RMSprop optimizer with learning rate "learning_rate" and compile them with a suitable loss function and accuracy as a metric. Then instantiate a MyCallback object with the
MNIST test data set "x_test_mnist", "y_test_mnist" and the FashionMNIST test data set "x_test_fmnist", "y_fmnist". Finally train the model first on the MNIST training data set and then on the FashionMNIST data set.
Hint: Use the MyCallback object "my_cb" during the training in order to record the accuracies on both data sets during training. """

extended_mdl = MyExtendedModel()
extended_opt = k.optimizers.RMSprop(learning_rate=learning_rate)
extended_mdl.compile(loss="sparse_categorical_crossentropy", optimizer=extended_opt, metrics=["accuracy"] )

my_cb = MyCallback(x_test_mnist, y_test_mnist, x_test_fmnist, y_test_fmnist, batch_size = batch_size)

extended_mdl.fit(x_train_mnist, y_train_mnist, batch_size=batch_size, callbacks =my_cb, epochs =epochs) # Train on MNIST
extended_mdl.fit(x_train_fmnist,y_train_fmnist, batch_size=batch_size, callbacks =my_cb, epochs =epochs) # Train on FashionMNIST
