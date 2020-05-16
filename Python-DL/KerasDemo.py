#%% Import Keras
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers


#%% Do Model
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train.reshape(60000,784).astype('float32')
x_test=x_test.reshape(10000,784).astype('float32')
y_train=keras.utils.to_categorical(y_train,num_classes=10)
y_test=keras.utils.to_categorical(y_test,num_classes=10)
model=Sequential()
model.add(Dense(1024,input_shape=(784,),activation='tanh'))
model.add(Dense(10,activation='softmax'))
sgd=optimizers.SGD(lr=0.1,momentum=0.0,decay=0.0, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
history=model.fit(x_train,y_train,batch_size=128, epochs=60,verbose=1,validation_data=(x_test,y_test))
#%% Next cell
