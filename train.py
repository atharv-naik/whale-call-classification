'''
train.py
    This script is used to train and tune the CNN model. The model is trained on the preprocessed data and the best hyperparameters

Usage:
    python training.py'''

# Train Convolutional Neural Network for blue whale A-call recognition 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import pandas as pd
import keras
from joblib import dump, load
from keras.optimizers import SGD, Adam, Adagrad, RMSprop
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import make_scorer, f1_score, log_loss
from keras.models import Sequential
import keras.backend as K
K.set_image_data_format('channels_last')
import warnings
# Suppress warnings
warnings.filterwarnings('ignore')


"""
A spectrogram was obtained from every audio file in the training set The spectrogram’s contrast was enhanced hori-
zontally (temporal dimension) and vertically (frequency dimension) by removing extreme values and implementing
a sliding mean.
• The CNN model’s hyperparameters were optimized using 3-Fold Cross Validation via GridSearchCV. The CNN
was fit using the enhanced training set
• .If the vertically-enhanced input yielded 1 and the horizontally-enhanced input yielded 0, the final predicted label
was 1.
• We choosed The Receiver Operating Characteristic (ROC) for evaluation, being a measure of the true positive rate vs
false positive rate as the discrimination threshold of the binary classifier is varied. The area under the curve (AUC)
is a single number metric of a binary classifier’s ROC curve and it is this ROC-AUC score that is used for evaluation
of the CNN model.
"""


# Load the data
print("Loading data...")
data = np.load('training_data_preprocessed.npz')
X_train, Y_train = data.files
X_train = data[X_train]
Y_train = data[Y_train]


class CNNModel(object):
    def __init__(self, learning_rate=0.01, activation='relu', optimizer=SGD):
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer
        self.classes_ = [0, 1]

    def create_model(self):
        model = Sequential()
        # Dropout on the visible layer (1 in 5 probability of dropout) 
        model.add(Dropout(0.2,input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3]),name='drop1'))
        # Conv2D -> BatchNorm -> Relu Activation -> MaxPooling2D
        model.add(Conv2D(15,(7,7),strides=(1,1),name='conv1'))
        model.add(BatchNormalization(axis=3,name='bn1'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2,2),name='max_pool1'))
        # Conv2D -> BatchNorm -> Relu Activation -> MaxPooling2D
        model.add(Conv2D(30,(7,7),strides=(1,1),name='conv2'))
        model.add(BatchNormalization(axis=3,name='bn2'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2,2),name='max_pool2'))
        # Flatten to yield input for the fully connected layer 
        model.add(Flatten())
        model.add(Dense(200,activation='relu',name='fc1'))
        # Dropout on the fully connected layer (1 in 2 probability of dropout) 
        model.add(Dropout(0.5,name='drop2'))
        # Single unit output layer with sigmoid nonlinearity 
        model.add(Dense(1,activation='sigmoid',name='fc2'))
        # Use Stochastic Gradient Discent for optimization 
        sgd = SGD(learning_rate=0.1,decay=0.005,nesterov=False)
        model.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['accuracy'])
        return model

    def fit(self, X, y, batch_size=128, epochs=10):
        self.model = self.create_model()
        self.model.fit(X, y, batch_size=batch_size, epochs=epochs)

    def predict_proba(self, X):
        y_pred = self.model.predict(X)
        return np.column_stack((1 - y_pred, y_pred))

    def predict(self, X):
        return np.round(self.model.predict(X))

    def get_params(self, deep=True):
        return {'learning_rate': self.learning_rate, 'activation': self.activation}

    def set_params(self, **params):
        self.learning_rate = params['learning_rate']
        self.activation = params['activation']
        self.optimizer = params['optimizer']
        return self
     

def main():
    # Define the hyperparameters to search
    param_dist = {'learning_rate': [0.1, 0.01],
                'batch_size': [128, 2*128, 3*128],
                'epochs': [20, 30, 40],
                'optimizer': [SGD, Adam, Adagrad, RMSprop],
                'activation': ['tanh', 'relu', 'sigmoid']}


    # Create the scoring function using the f1_score
    f1_scorer = make_scorer(f1_score, average='micro')

    # Create the GridSearch object
    grid_search = GridSearchCV(CNNModel(), param_grid=param_dist, scoring=f1_scorer, cv=2, verbose=3, return_train_score=True)
    # Fit the randomized search object to the training data
    print('Fitting model...')
    grid_result = grid_search.fit(X_train, Y_train)

    print('Best: %f using %s' % (grid_result.best_score_,grid_result.best_params_))
    # Print mean and standard deviation of accuracy scores for each combination
    # of parameters evaluated by GridSearchCV
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean,std,param in zip(means,stds,params):
        print('%f (%f) with: %r' % (mean,std,param))

    # save the best model
    print('Saving the best model...')
    best_model = grid_result.best_estimator_
    dump(best_model, 'best_model.joblib')


if __name__ == '__main__':
    main()
