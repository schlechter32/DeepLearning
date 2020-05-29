#%% Imports
import tarfile
import os
import glob
import tensorflow as tf
import tensorflow.keras as k
import matplotlib.pyplot as plt
from keras import optimizers
#from keras.models import Sequential
import numpy as np
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.random.set_seed(0)
np.random.seed(0)
#%% load list of filenames
(os.path.isdir("data\\UTKFace"))
def load_file_names():
    # Check if data has been extracted and if not extract it
    if (os.path.isdir("data/UTKFace")):
        print("Data set already extracted...")
    else:
        print("Extracting data set...")
        tar = tarfile.open("./data/UTKFace.tar.gz")
        tar.extractall("./data")
        tar.close()

    # Get a list of all files in data set
    files = glob.glob("./data/UTKFace/*.jpg")
    files
    labels = [int(f_name.split("\\")[-1].split("_")[0]) for f_name in files]
    labels
    return files, labels
#%% Work with list
files, labels = load_file_names()
files[0]
#img = plt.imread(files[0])
# plt.imshow(img)
# plt.title("Age: {}".format(labels[0]))
# plt.axis("off")
# plt.show()
#%% Make Model
""" Implement a small CNN with two convolutional and three dense layers. The conv. layers should have 8 and 16 filters of size 5x5,
astride of 4 and a ReLU activation. Also create a flatten layer and a dropout layer
with 0.2 droprate. The dense layers should have 128, 64 and 1 neurons and again a ReLU activation.
For implementing the layers use predefined layer classes of Keras, e.g. k.layers.Conv2D or k.layers.Dense.
Hint: Create the layer objects in the constructor and call them in the right order for implementing the forward pass.
You need to pass the "training" argument to the dropout layer in order to activate dropout during
training and deactivate it during testing. """
#%% Different way for testing
#%% Make model class
class MyModel(k.Model):
    def __init__(self):
        # Create layers
        super(MyModel, self).__init__()
        self.conv0 = k.layers.Conv2D(8,kernel_size=(5,5),activation = 'relu', kernel_initializer='glorot_uniform',bias_initializer='zeros',  strides=4,data_format= 'channels_last')
        self.conv1 =  k.layers.Conv2D(16,kernel_size=(5,5),activation = 'relu', kernel_initializer='glorot_uniform',bias_initializer='zeros' ,strides=4, data_format= 'channels_last')
        self.flatten = k.layers.Flatten()
        self.dropout = k.layers.Dropout(0.2)
        self.dense0 = k.layers.Dense(128, activation ='relu' ,kernel_initializer='glorot_uniform', bias_initializer = 'zeros')
        self.dense1 = k.layers.Dense(64, activation ='relu' ,kernel_initializer='glorot_uniform', bias_initializer = 'zeros')
        self.dense2 = k.layers.Dense(1, activation ='relu',kernel_initializer='glorot_uniform', bias_initializer = 'zeros')

    def call(self, inputs, training = False):
        # Implement forward pass
        #print (inputs.shape)
        output = self.conv0(inputs)
        #print(output.shape)
        output = self.conv1(output)
        output = self.flatten(output)
        output = self.dropout(output, training)
        output = self.dense0(output)
        output = self.dense1(output)
        output = self.dense2(output)
        return output
#%% Test model building
# mdl = MyModel()
# mdl.build((batch_size, 200,200,3))
# for image,label in image_ds:
#     mdl.__call__(image)
#%% General Parameters
N_epochs = 20
learning_rate = 0.001
batch_size = 64
N_training_examples = 20000
N_validation_examples = 4*batch_size

N_parallel_iterations = 4
N_prefetch = 8
N_shuffle_buffer = 20000
#%% Get Data ready and create datasets
""" Since we only have the file name and labels of the images, we need to actually load an image into system memory
if it is needed. For this we will define a function that parses the image, normalizes it and reshapes
the label. This reshaping is a technical detail that avoids unintended behaivior during the calculation of the loss,
i.e. it avoids unintended broadcasting. Your task is to fill in the missing code for loading,
decoding and normalizing the image. In this example we normalize the image to pixels in the range between 0 and 1.
Hint: For reading and decoding the image use the functions defined in tf.io. The images are stored in the
JPEG format."""

def parse_func(filename, label):
    image_string = tf.io.read_file(filename)# Read the image
    image_decoded = tf.io.decode_jpeg(image_string, channels=3)# Decode the image
    image = tf.cast(image_decoded, tf.float32) / 255.0 # Normalize the image
    label = tf.expand_dims(tf.cast(label, tf.float32), axis=-1)
    return image , label

""" We now build a tensorflow Dataset object that shuffles the data with a shuffle buffer of size "N_shuffle_buffer", applies the parse_func via the .map() function with "N_parallel_iterations", creates batches
of size "batch_size" and prefetches with "N_prefetch". Please fill in the missing code. """
 #%% build dataset
def build_dataset(files, labels, batch_size):
    # Create tf data set
    ds=tf.data.Dataset.from_tensor_slices((files,labels))
    # Create data set of files and labels
    ds = ds.shuffle(N_shuffle_buffer)# Enable shuffling
    ds = ds.map(parse_func, num_parallel_calls=N_parallel_iterations )# Apply parse_func
    ds = ds.batch(batch_size).prefetch(buffer_size = N_prefetch) # Batch and prefetch
    return ds

# Shuffle data and labels
train_ds = build_dataset(files[0:N_training_examples],
                                     labels[0:N_training_examples], batch_size).repeat()
#train_ds.shape()
validation_ds = build_dataset(files[N_training_examples:N_training_examples+N_validation_examples],
                                          labels[N_training_examples:N_training_examples+N_validation_examples],
                                          batch_size)
validation_ds
test_ds = build_dataset(files[N_training_examples+N_validation_examples:],
                                    labels[N_training_examples+N_validation_examples:], batch_size)
#%% Loss and training
""" For this application we will use the Mean Absolute Error (MAE) as a loss function for training. Please implement the loss and the training step."""

# Define loss function
def loss(y, y_pred):
    return tf.keras.losses.MeanAbsoluteError()(y, y_pred)



# Define training step as a complete graph
@tf.function
def train_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        y_pred = mdl.__call__(x,training = True)# Predict with model on "x"
        loss_val = loss(y,y_pred)
    grads = tape.gradient(loss_val, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_val
#%% Set up model
mdl = MyModel()
opt = tf.optimizers.RMSprop(learning_rate)
mdl.build((batch_size, 200, 200, 3))
mdl.summary()
#%% Run Training
# Run training
epoch = 0
train_loss = 0.0
train_iters = 0
for train_images, train_labels in train_ds:
    train_loss += train_step(mdl, opt, train_images, train_labels)# Perform a train step
    train_iters += 1
    if train_iters ==int(N_training_examples/batch_size): # An epoch is completed 312
        epoch += 1
        val_loss = 0.0
        val_iters = 0
        for val_images, val_labels in validation_ds:
            y_pred = mdl.__call__(val_images)# Predict on validation images
            loss_val = loss(val_labels, y_pred) # Compute loss for validation
            val_loss += loss_val
            val_iters += 1
            if val_iters == int(N_validation_examples/batch_size):
                print('--------------------------------------------------')
                print("Epoch: {} Training loss: {:.5} Validation loss {:.5}"
                      .format(epoch, train_loss/train_iters, val_loss/val_iters))
                print('--------------------------------------------------')
                break
        train_loss = 0.0
        train_iters = 0
    if epoch == N_epochs:
        break
for x_t, y_t in test_ds:
    y_pred = mdl.__call__(x_t, training = False)
    y_t =tf.reshape(y_t, [-1,1]) # Compute a prediction with "mdl" on the input "x_t"
    test_loss = loss(y_t, y_pred)# Compute the Mean Squared Error (MSE) for the prediction "y_pred" and the targets "y_t"
print("Test loss: {:.5}".format(test_loss))

#%% Use dataset to estimate age
""" For predicting on the uploaded image, open and decode it using tf.io. After that we also need to normalize and resize it using tf.image.resize. Hint: For predicting with our model we need to add a batch dimension
of 1, since we effectively feed our model with a batch containing only one image."""
fn = files[550]
labels[550]
# Load and predict on an image
image_string = image_string = tf.io.read_file(fn)# Load image with path "fn"
image_decoded =tf.expand_dims(tf.io.decode_jpeg(image_string, channels=3), axis= 0)  # Decode the image and add a batch dimension
image =tf.cast(image_decoded, tf.float32) / 255.0 # Normalize, resize to 200x200 pixels and cast image to tf.float32
age = mdl.__call__(image)

# Plot image and prediction
plt.imshow(np.squeeze(image.numpy()))
plt.title("Age prediction: {:.3}".format(np.squeeze(age.numpy())))
plt.show()
#%% save model
mdl.save(filepath = 'AgeModel/One')
model = k.models.load_model('AgeModel/One')
#%% Test loaded model
fn = files[550]
labels[550]
# Load and predict on an image
image_string = image_string = tf.io.read_file(fn)# Load image with path "fn"
image_decoded =tf.expand_dims(tf.io.decode_jpeg(image_string, channels=3), axis= 0)  # Decode the image and add a batch dimension
image =tf.cast(image_decoded, tf.float32) / 255.0 # Normalize, resize to 200x200 pixels and cast image to tf.float32
age = model.__call__(image)

# Plot image and prediction
plt.imshow(np.squeeze(image.numpy()))
plt.title("Age prediction: {:.3}".format(np.squeeze(age.numpy())))
plt.show()
