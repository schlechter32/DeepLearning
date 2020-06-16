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
# Define constants
batch_size = 128
epochs = 20
learning_rate = 0.001

# (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()
# x_train_mnist = np.expand_dims(x_train_mnist, axis=-1).astype(np.float32)
# x_test_mnist = np.expand_dims(x_test_mnist, axis=-1).astype(np.float32)
# # y_train_mnist=k.utils.to_categorical(y_train_mnist,num_classes=10)
# # y_test_mnist=k.utils.to_categorical(y_test_mnist,num_classes=10)
# y_train_mnist[0]
# #%% Define model
# """ Define a small convolutional network for classification with two conv. and 3 dense layers. The conv. layers should have 8/16 filters a kernel size of 3x3 and a stride of 2. The dense layers should have 128/64/?
# neurons. Choose the activation functions of the layers accordingly."""
# class MyModel(k.Model):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         # Layer definition
#         super(MyModel, self).__init__()
#         self.conv0 = k.layers.Conv2D(8,kernel_size=(3,3),activation = 'relu', kernel_initializer='glorot_uniform',bias_initializer='zeros',  strides=2)
#         self.conv1 =  k.layers.Conv2D(16,kernel_size=(3,3),activation = 'relu', kernel_initializer='glorot_uniform',bias_initializer='zeros' ,strides=2)
#         self.flatten = k.layers.Flatten()
#         #self.dropout = k.layers.Dropout(0.2)
#         self.dense0 = k.layers.Dense(128, activation ='relu' ,kernel_initializer='glorot_uniform', bias_initializer = 'zeros')
#         self.dense1 = k.layers.Dense(64, activation ='relu' ,kernel_initializer='glorot_uniform', bias_initializer = 'zeros')
#         self.dense2 = k.layers.Dense(10, activation ='softmax',kernel_initializer='glorot_uniform', bias_initializer = 'zeros')
#
#
#     def call(self, inputs, training=False):
#         # Call layers appropriately to implement a forward pass
#         output = self.conv0(inputs)
#         #print(output.shape)
#         output = self.conv1(output)
#         output = self.flatten(output)
#         #output = self.dropout(output, training)
#         output = self.dense0(output)
#         output = self.dense1(output)
#         output = self.dense2(output)
#         return output
# #%% Instantiate model
# """ Instantiate an object of MyModel and an RMSprop optimizer with learning rate given by the constant "learning_rate". Compile the model with a suitable loss function and add accuracy as a metric. """
#
# mdl = MyModel()
# opt = k.optimizers.RMSprop(learning_rate = learning_rate)
# mdl.compile(opt, loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
# #%% Training step
# """ Train the model mdl on the training data with a batch size of "batch_size" for "epochs" epochs. Train with 10% of the training data as validation data. """
# history_no_dropout = mdl.fit(x = x_train_mnist[0:54000], y= y_train_mnist[0:54000], batch_size=batch_size, epochs=epochs, validation_data=(x_train_mnist[54000:60000], y_train_mnist[54000:60000]))
# #%% Visualize training
# plt.plot(history_no_dropout.history["loss"])
# plt.plot(history_no_dropout.history["val_loss"])
# plt.legend(["loss", "val_loss"])
# plt.xticks(range(epochs))
# plt.xlabel("epochs")
# plt.title("Training process")
# plt.show()
# #%% Define Model with dropouzt layer
# """ Create a new model, which is identical to MyModel execpt for a dropout layer between the conv. and dense layers. As a dropout rate use 0.25 """
#
# class MyDropoutModel(k.Model):
#     def __init__(self):
#         super(MyDropoutModel, self).__init__()
#             # Layer definition
#         self.conv0 = k.layers.Conv2D(8,kernel_size=(3,3),activation = 'relu', kernel_initializer='glorot_uniform',bias_initializer='zeros',  strides=2)
#         self.conv1 =  k.layers.Conv2D(16,kernel_size=(3,3),activation = 'relu', kernel_initializer='glorot_uniform',bias_initializer='zeros' ,strides=2)
#         self.flatten = k.layers.Flatten()
#         self.dropout = k.layers.Dropout(0.25)
#         self.dense0 = k.layers.Dense(128, activation ='relu' ,kernel_initializer='glorot_uniform', bias_initializer = 'zeros')
#         self.dense1 = k.layers.Dense(64, activation ='relu' ,kernel_initializer='glorot_uniform', bias_initializer = 'zeros')
#         self.dense2 = k.layers.Dense(10, activation ='softmax',kernel_initializer='glorot_uniform', bias_initializer = 'zeros')
#
#
#     def call(self, inputs, training=False):
#         # Call layers appropriately to implement a forward pass
#         output = self.conv0(inputs)
#         #print(output.shape)
#         output = self.conv1(output)
#         output = self.flatten(output)
#         output = self.dropout(output, training)
#         output = self.dense0(output)
#         output = self.dense1(output)
#         output = self.dense2(output)
#         return output
# #%% Use dropout model
# """ Instantiate a MyDropoutModel object, compile and train it on the training data. Use the same optimizer and parameters for training as before. """
#
# dropout_mdl = MyDropoutModel()
# droput_opt = k.optimizers.RMSprop(learning_rate = learning_rate)
# dropout_mdl.compile(opt, loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
#
# history_dropout = dropout_mdl.fit(x = x_train_mnist[0:54000], y= y_train_mnist[0:54000], batch_size=batch_size, epochs=epochs, validation_data=(x_train_mnist[54000:60000], y_train_mnist[54000:60000]))
#
# plt.plot(history_no_dropout.history["loss"])
# plt.plot(history_no_dropout.history["val_loss"])
# plt.plot(history_dropout.history["loss"])
# plt.plot(history_dropout.history["val_loss"])
# plt.legend(["loss", "val_loss", "loss w. dropout", "val_loss w. dropout"])
# plt.xticks(range(epochs))
# plt.xlabel("epochs")
# plt.title("Training process")
# plt.show()
# #%% End of file
# print('Done Boss')
#%%Transfer learning part of the excercise
#wget.download('http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz')
#!tar -xzf 101_ObjectCategories.tar.gz
#%% Image augmentation
N_samples_Caltech101 = 9247
val_split = 0.1
datagen = k.preprocessing.image.ImageDataGenerator(validation_split=val_split,
                                                   preprocessing_function=k.applications.mobilenet_v2.preprocess_input,
                                                   rotation_range=20,
                                                   width_shift_range=0.1,
                                                   height_shift_range=0.1,
                                                   shear_range=0.1,
                                                   zoom_range=0.1,
                                                   horizontal_flip=True)
#%% Obtain pretrained model
""" Instantiate a MobileNetV2 with weights pretrained on ImageNet and without the top/output layer. Hint: Use Keras Applications"""

base_model = k.applications.mobilenet_v2.MobileNetV2(include_top = False)
base_model.summary()
""" Create a transfer learning model, that uses a pretrained model "pretrained_model" and appends a 2D global average pooling layer, a dropout layer (droprate 0.25) and a dense output layer with 102 neurons. """

class MyTransferModel(k.Model):
    def __init__(self, pretrained_model):
        super(MyTransferModel, self).__init__()
        self.pretrained_model = pretrained_model
        # self.model.add(k.layers.GlobalAvgPool2D())
        # self.model.add(k.layers.Dropout(0.25))
        # self.model.add(k.layers.Dense(102, activation ='softmax',kernel_initializer='glorot_uniform', bias_initializer = 'zeros'))
        self.glob_pool = k.layers.GlobalAvgPool2D()
        self.dropout = k.layers.Dropout(0.25)
        self.dense_out = k.layers.Dense(102, activation ='softmax',kernel_initializer='glorot_uniform', bias_initializer = 'zeros')
    def call(self, inputs, training=False):
        # Call pretrained model and layers appropriately to implement forward pass
        output = self.pretrained_model(inputs)
        output = self.glob_pool(output)
        output = self.dropout(output,training)
        output = self.dense_out(output)
        return output
#%% Initialize transfer model
""" Instantiate a MyTransferModel object and a RMSprop optimizer, compile them with a suitable loss and accuracy as a metric. Use "base_model" as the pretrained model. """

tf_batch_size = 32
tf_epochs = 10
tf_learning_rate = 0.001
# opt = k.optimizers.RMSprop(learning_rate = learning_rate)
# mdl.compile(opt, loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
tf_mdl =MyTransferModel(base_model)
tf_opt =k.optimizers.RMSprop(learning_rate = learning_rate)
base_model.trainable = False
tf_mdl.compile(tf_opt, loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
tf_mdl.build((tf_batch_size, 224, 224, 3))
tf_mdl.summary()
#%% Train awesome new transfer model
""" Create the data set generators and train the model for "tf_epochs" epochs. Hint: Use the class mode "sparse" and the appropriate subsets for creating the generators and
steps_per_epoch=int((1.0-val_split)*N_samples_Caltech101/tf_batch_size) as well as a suitable number of validation_steps for the fit function. The data for the Caltech101
data set is located in the "101_ObjectCategories" directory. """
#val_split = 0.9
steps_per_epoch=int((1.0-val_split)*N_samples_Caltech101/tf_batch_size)
print(steps_per_epoch)
train_gen = datagen.flow_from_directory("101_ObjectCategories", class_mode ='sparse' )

val_gen = datagen.flow_from_directory("101_ObjectCategories", class_mode ='sparse')
tf_history_0 = tf_mdl.fit(train_gen, validation_data = val_gen, validation_steps = int(steps_per_epoch *0.4) ,epochs = tf_epochs,  steps_per_epoch = steps_per_epoch)



# dropout_mdl.fit(x = x_train_mnist[0:54000], y= y_train_mnist[0:54000], batch_size=batch_size, epochs=epochs, validation_data=(x_train_mnist[54000:60000], y_train_mnist[54000:60000]))
#%% Different learning rate
""" Reinstantiate the RMSprop optimizer with the changed learning rate and set the base_model to be trainable. Then compile it with the newly instantiate optimizer, a suitable loss function and accuracy as a metric
and continue training on the Caltech101 data set. Hint: Angain use steps_per_epoch=int((1.0-val_split)*N_samples_Caltech101/tf_batch_size) as well as a suitable number of validation_steps to train the model fo
"tf_epochs" epochs on the data set. """

tf_learning_rate = 0.00001
tf_opt = tf_opt =k.optimizers.RMSprop(learning_rate = learning_rate)
# Set base_model to be trainable
base_model.trainable = True
tf_mdl.compile(loss="sparse_categorical_crossentropy", optimizer=tf_opt, metrics=["accuracy"])
tf_mdl.build((tf_batch_size, 224, 224, 3))
tf_mdl.summary()

tf_history_1 = tf_mdl.fit(train_gen, validation_data = val_gen, validation_steps = int(steps_per_epoch *0.4) ,epochs = tf_epochs,  steps_per_epoch = steps_per_epoch)
#%% Visualize
plt.plot(tf_history_0.history["accuracy"]+tf_history_1.history["accuracy"])
plt.plot(tf_history_0.history["val_accuracy"]+tf_history_1.history["val_accuracy"])
plt.xticks(range(len(tf_history_0.history["accuracy"]+tf_history_1.history["accuracy"])))
plt.axvline(9, color="green")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Acc", "Val. Acc", "Start fine tuning"])
plt.show()
#%% Evaluate transfer learning
""" Instantiate a new base_model, now with randomly initialized weights, and instantiate a new MyTransferModel object. Compile it with a RMSprop optimizer, a suitable loss and accuracy as a metric.  """

tf_batch_size = 32
tf_epochs = 5
tf_learning_rate = 0.001
base_model_rand_init = k.applications.mobilenet_v2.MobileNetV2(include_top = False, weights = None)# Instantiate a randomly initialized MobileNetV2 without it's top/output layer
tf_mdl = MyTransferModel(base_model_rand_init)
tf_opt =k.optimizers.RMSprop(learning_rate = learning_rate)
tf_mdl.compile()
tf_mdl.build((tf_batch_size, 224, 224, 3))
tf_mdl.summary()

""" Train this newly instantiated model on the Caltech101 data set. Hint: Again use steps_per_epoch=int((1.0-val_split)*N_samples_Caltech101/tf_batch_size) as well as a suitable number of validation_steps. """

tf_history_2 = tf_mdl.fit(train_gen, validation_data = val_gen, validation_steps = int(steps_per_epoch *0.4) ,epochs = tf_epochs,  steps_per_epoch = steps_per_epoch)
#%% Catastrophic forgetting explained
""" Shift the labels of the FashionMNIST data set to {10,...,19}. """

(x_train_fmnist, y_train_fmnist), (x_test_fmnist, y_test_fmnist) = tf.keras.datasets.fashion_mnist.load_data()

x_train_fmnist = np.expand_dims(x_train_fmnist, axis=-1).astype(np.float32)
y_train_fmnist = y_train_fmnist + 10# Shift labels of training data
x_test_fmnist = np.expand_dims(x_test_fmnist, axis=-1).astype(np.float32)
y_test_fmnist = # Shift labels of testing data
#%% Define Model for Catastrophic forgetting
""" Implement a new model, which is capable of classifying 20 classes. Use two conv. layers with 8/16 filters of size 3x3 and a stride of 2, a dropout layer between the conv. and dense layers with a droprate of 0.25
three dense layers with 128/64/? neurons. Choose all activation functions appropriately."""

class MyExtendedModel(k.Model):
    def __init__(self):
        super(MyExtendedModel, self).__init__()
    #Layer definition
        self.conv0 = k.layers.Conv2D(8,kernel_size=(3,3),activation = 'relu', kernel_initializer='glorot_uniform',bias_initializer='zeros',  strides=2)
        self.conv1 =  k.layers.Conv2D(16,kernel_size=(3,3),activation = 'relu', kernel_initializer='glorot_uniform',bias_initializer='zeros' ,strides=2)
        self.flatten = k.layers.Flatten()
        self.dropout = k.layers.Dropout(0.25)
        self.dense0 = k.layers.Dense(128, activation ='relu' ,kernel_initializer='glorot_uniform', bias_initializer = 'zeros')
        self.dense1 = k.layers.Dense(64, activation ='relu' ,kernel_initializer='glorot_uniform', bias_initializer = 'zeros')
        self.dense2 = k.layers.Dense(20, activation ='softmax',kernel_initializer='glorot_uniform', bias_initializer = 'zeros')


    def call(self, inputs, training=False):
        # Call layers appropriately to implement a forward pass
        output = self.conv0(inputs)
        #print(output.shape)
        output = self.conv1(output)
        output = self.flatten(output)
        output = self.dropout(output, training)
        output = self.dense0(output)
        output = self.dense1(output)
        output = self.dense2(output)
        return output
