# soachishti (p146011@nu.edu.pk)

import os
import sys
import random

import numpy as np
from scipy import misc
from scipy.ndimage import imread
from sklearn.neural_network import MLPClassifier
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

from keras.callbacks import EarlyStopping

size_of_input = 10 * 10 # Size of our image
num_classes = 10
total_accuracy = []
verbose = 1

# Setting for Keras and Sklearn

# Culprit Settings START
momentum = 0.9 #0.9
nesterov = True # Keras
shuffle  = True # Keras
# Culprit Settings END

early_stopping = False
batch_size = 64
neurons = 10                       # Number of neuron for hidden layer
activation_keras = 'sigmoid'        # Keras
activation_sklearn = 'logistic'     # Sklearn
output_activation = 'softmax'       # Keras   
epochs = 100                        # Keras
max_iter = 100                      # Sklearn 
learning_rate = 0.01

loss_function = 'categorical_crossentropy' # 'kullback_leibler_divergence'

def load_data(folder='test-set-519', train_percent=0.8):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for number in range(10):
        DIR = folder + "/" + str(number) + "/"
        data_count = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
        train_count = int(data_count * train_percent)
        
        # Get unique selection for each images.
        random_selection = random.sample(range(data_count), data_count)

        for i in random_selection:
            path = folder + "/" + str(number) + "/" + str(i) + ".bmp"
            img = misc.imread(path) # / 255
            if train_count > 0:
                x_train.append(img)
                y_train.append(number)
                train_count -= 1
            else:
                x_test.append(img)
                y_test.append(number)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    x_train = x_train.reshape(len(x_train), 100) # Multiply input with 10
    x_test = x_test.reshape(len(x_test), 100)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return x_train, y_train, x_test, y_test

def using_sklearn(x_train, y_train, x_test, y_test):
    print ("\n###### Using Sklearn ######")

    mlp = MLPClassifier(hidden_layer_sizes=(neurons,), activation=activation_sklearn,early_stopping=early_stopping, solver='sgd', max_iter = max_iter, batch_size=batch_size, learning_rate_init=learning_rate)
    mlp.fit(x_train,y_train)

    print('Number of samples in training set: %d, number of samples in test set: %d'%(len(y_train), len(y_test)))
    
    score_train = mlp.score(x_train, y_train)
    score_test  = mlp.score(x_test, y_test)

    print ('Train Accuracy:', score_train)
    print ('Test Accuracy:', score_test)
    print ('Layers:', mlp.n_layers_)
    print ('Output Layer size: ',mlp.n_outputs_)
    print ('Number of Iteration: ',mlp.n_iter_)
    print ('Output Activation: ',mlp.out_activation_)
    print ('Loss', mlp.loss_)

def using_keras(x_train, y_train, x_test, y_test):
    print ("\n\n###### Using Keras ######")
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test  = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
        
    model.add(Dense(neurons, activation=activation_keras, input_shape=(size_of_input,)))
    model.add(Dense(num_classes, activation=output_activation))

    model.compile(loss=loss_function, 
                  optimizer=SGD(lr=learning_rate, momentum=momentum, nesterov=nesterov),
                  metrics=['accuracy'])
    

    callbacks = []

    if early_stopping == True:
        callbacks.append(EarlyStopping(monitor='loss', min_delta=0e-4, patience=2, verbose=verbose, mode='auto'))

    model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    shuffle=shuffle,
                    callbacks=callbacks,
                    verbose=verbose
                )
    
    score_test = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=verbose)
    score_train = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=verbose)
    
    print('\nNumber of samples in training set: %d, number of samples in test set: %d'%(len(y_train), len(y_test)))

    print ('Train accuracy:', score_train[1])
    print ('Test accuracy:', score_test[1])
    print ('Layers:', 1 + 1 + 1)
    print ('Output Layer size: ', num_classes)
    print ('Number of Epochs: ', epochs)
    print ('Output Activation: ', output_activation)
    #print ('Loss', model.)


(x_train, y_train, x_test, y_test) = load_data()
using_sklearn(x_train, y_train, x_test, y_test)
using_keras(x_train, y_train, x_test, y_test)

# Error Function Comparision for Keras (We cannot change error function in Sklearn)
#for l in ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'categorical_hinge', 'logcosh', 'categorical_crossentropy', 'sparse_categorical_crossentropy', 'kullback_leibler_divergence', 'poisson', 'cosine_proximity']: 
#    loss_function = l 
#    print ("\n\n\nLoss: " + loss_function)
#    try:
#        using_keras(x_train, y_train, x_test, y_test)
#    except:
#        print ("Unexpected error:", sys.exc_info()[0])
#        pass7