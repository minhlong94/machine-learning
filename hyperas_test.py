from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.models import Sequential
from hyperas import optim
from hyperas.distributions import choice, uniform
import numpy as np
from sklearn.metrics import roc_auc_score

def custom_get_data():
    """
    Data providing function:
    """
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test



def create_model(x_train, y_train, x_test, y_test):
    
    model = Sequential()
    model.add(Dense(784, input_shape=(784,), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(392, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(42, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))
    model.compile(loss='categorical_crossentropy',
                    optimizer={{choice(["rmsprop", "adam", "sgd"])}}, metrics=[tf.keras.metrics.AUC()])
        
    model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2, validation_split=0.2)
    predictions = model.predict(x_test, batch_size=32, verbose=0)
    roc_acc = roc_auc_score(y_test, predictions)
    print("ROC: {}".format(roc_acc))
    return {"loss": -roc_acc, "status": STATUS_OK, "model": model}

if __name__ == '__main__':
    best_run,best_model = optim.minimize(model=create_model, data=custom_get_data,algo=tpe.suggest, 
                                        max_evals=10, trials=Trials()) # use this if run .py file
    x_train, y_train, x_test, y_test = custom_get_data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(x_test, y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
