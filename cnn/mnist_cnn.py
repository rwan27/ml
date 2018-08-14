# See: https://yashk2810.github.io/Applying-Convolutional-Neural-Network-on-the-MNIST-dataset/
# See: https://github.com/yashk2810/MNIST-Keras/blob/master/Notebook/MNIST_keras_CNN-99.55%25.ipynb
# See: https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
#from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
#from tensorflow.keras.layers import LeakyReLU 
#from tensorflow.keras.layers import BatchNormalization

np.random.seed(25)

#
# LOAD MNIST
#
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)
print("X_test original shape", X_test.shape)
print("y_test original shape", y_test.shape)

#plt.imshow(X_train[69], cmap='gray')
#plt.title('Class '+ str(y_train[69]))
#plt.show()

#
# TRANFORM DATA TO FLOAT AND NORMALIZE
#
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train/=255
X_test/=255
print("X_train original shape", X_train.shape)
print("X_test original shape", X_test.shape)

number_of_classes = 10
Y_train = to_categorical(y_train, number_of_classes) # 60K elements, each is a 1D vector of 10 classes - a List of Lists
Y_test = to_categorical(y_test, number_of_classes) # 10K elements, each is a 1D vector of 10 classes - a List of Lists
print(y_train[69]) # scalar 0 for element 69
print(Y_train[69]) # converted scalar in List in '10 class form'
Y_test_sum = sum(Y_test)
Y_train_sum = sum(Y_train)
print(Y_train_sum) # freq
print(sum(Y_train_sum))
print(Y_test_sum) # freq
print(sum(Y_test_sum))

# Three steps to Convolution
# 1. Convolution
# 2. Activation
# 3. Polling
# Repeat Steps 1,2,3 for adding more hidden layers
#
# 4. After that make a fully connected network
# This fully connected network gives ability to the CNN to classify the samples

network = models.Sequential()
network.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))

layers.BatchNormalization(axis=-1)
network.add(layers.Conv2D(32, (3, 3), activation='relu'))
network.add(layers.MaxPooling2D(pool_size=(2,2)))

layers.BatchNormalization(axis=-1)
network.add(layers.Conv2D(64,(3, 3), activation='relu'))

layers.BatchNormalization(axis=-1)
network.add(layers.Conv2D(64, (3, 3), activation='relu'))
network.add(layers.MaxPooling2D(pool_size=(2,2)))

network.add(layers.Flatten())

# Fully connected layer
layers.BatchNormalization()
network.add(layers.Dense(512, activation='relu'))
layers.BatchNormalization()
network.add(layers.Dropout(0.2))
network.add(layers.Dense(10, activation='softmax'))

network.summary()
network.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#To reduce over-fitting, we use another technique known as Data Augmentation. It rotates, shears, zooms, etc 
#the image so that the model learns to generalize and not remember specific data. If the model overfits, it will 
#perform very well on the images that it already knows but will fail if new images are given to it.
print("Data Augmentation")
gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)
test_gen = ImageDataGenerator()
train_generator = gen.flow(X_train, Y_train, batch_size=64)
test_generator = test_gen.flow(X_test, Y_test, batch_size=64)

#
# Train CNN using 'fit' - Fit model to the training data, with data generators
#
network.fit_generator(train_generator, 
	                  steps_per_epoch=60000//64, 
	                  epochs=5, 
                      validation_data=test_generator, 
                      validation_steps=10000//64)

test_loss, test_acc = network.evaluate(X_test, Y_test, verbose=2)
print('Test Loss =',test_loss,'Test accuracy =',test_acc)

'''
predictions = network.predict_classes(X_test)
predictions = list(predictions)
actuals = list(y_test)
sub = pd.DataFrame({'Actual': actuals, 'Predictions': predictions})
sub.to_csv('./output_cnn.csv', index=False)

#
#Pseudo Labelling
#

class MixIterator(object):
    def __init__(self, iters):
        self.iters = iters
        self.N = sum([it.n for it in self.iters])

    def reset(self):
        for it in self.iters: it.reset()

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        nexts = [next(it) for it in self.iters]
        n0 = np.concatenate([n[0] for n in nexts])
        n1 = np.concatenate([n[1] for n in nexts])
        return (n0, n1)

predictions = network.predict(X_test, batch_size=64)
predictions[:5]

batches = gen.flow(X_train, Y_train, batch_size=48)
test_batches = test_gen.flow(X_test, predictions, batch_size=16)

mi = MixIterator([batches, test_batches])
print(mi.N)

model.fit_generator(mi, steps_per_epoch=mi.N//64, epochs=5, validation_data=(X_test, Y_test))
'''

#saving model
print(">>> Saving model")
network.save('cnn_model.h5')
print(">>>Done")



