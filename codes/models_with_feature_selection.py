import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import math
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

trainset0 = pd.read_csv('Dataset 002_train data without normalization.csv')
testset0 = pd.read_csv('Dataset 002_test data without normalization.csv')
trainset1 = pd.read_csv('Dataset 002_train data after normalization.csv')
testset1 = pd.read_csv('Dataset 002_test data after normalization.csv')

def plotUnitLines(y_pred, startUnit, endUnit, y_test,  ncols=4, title='',figsize=[15, 15]):
    nunits = endUnit - startUnit + 1
    nrows = math.ceil(nunits/ncols)
    fig, axes=plt.subplots(nrows=nrows, ncols=ncols, clear=True, figsize=figsize)
    for unit in range(startUnit, endUnit+1):
        unitBooleanIndexes = testset0['unit']==unit;
        unit_pred = list(map(lambda x: x[0], y_pred[unitBooleanIndexes]))
        length = len(unit_pred)
        #Get the rul for each unit based on its unit boolean index
        y_rul = y_test[unitBooleanIndexes].values
        mse = mean_squared_error(y_rul, unit_pred)
        x = range(len(unit_pred))
        rowIndex = math.floor((unit-startUnit)/ncols)
        colIndex = (unit-startUnit)%ncols;
        ax = axes[rowIndex, colIndex]
        ax.plot(x, unit_pred, label= 'predicted')
        ax.plot(x, y_rul, label='actual')
        ax.set_title('Unit'+str(unit) + ', RBF: '+str(mse))
        ax.legend()
    fig.tight_layout()
    plt.show()

def plotLossHistory(model):
    plt.plot(model.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

def calculateScore(y_pred, y_test):
    y_pred = list(map(lambda x: x[0], y_pred))
    return mean_squared_error(y_pred, y_test)

def executeModel1(inputTrainset, inputTestset, columnsToDrop, numOfNodes=100, epochs=100, batch_size = None, verbose=True):
    print('==============================================================================')
    print ('Delete columns: ')
    print (columnsToDrop)
    print ('Num of ndoes: ' + str(numOfNodes))
    print ('Epochs: ' + str(epochs))
    print('==============================================================================')
    trainset = inputTrainset
    testset = inputTestset
    #Remove columns
    trainset = trainset.drop(columnsToDrop, axis = 1)
    testset = testset.drop(columnsToDrop, axis=1)
    #y_train, X_train
    y_train = trainset['RUL'].values
    X_train = trainset.drop('RUL', axis=1)
    #Converting to array
    X_train = X_train.values

    #y_test, X_test
    y_test0 = testset['RUL']
    y_test = y_test0.values
    X_test = testset.drop('RUL', axis=1)
    #Converting to array
    X_test = X_test.values
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Building the model
    import keras
    from keras.models import Sequential
    from keras.layers import Dense

    model = Sequential()
    layer1 = Dense(units=numOfNodes, activation='relu', input_dim = len(X_train[0]))
    layer2 = Dense(units=1, activation='relu')
    model.add(layer1)
    model.add(layer2)
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    #Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # #Predict
    # y_pred_model = model.predict(X_test)
    # print('THE MODEL SCORE IS: ' +str(calculateScore(y_pred_model, y_test)))
    # #plotUnitLines(y_pred_model, 1, 20, y_test0)

    #Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    #Predict
    y_pred_model = model.predict(X_test)
    print('THE MODEL TRAIN SCORE IS: ' +str(history.history['loss'][-1]) + ', TEST SCORE IS: ' +str(calculateScore(y_pred_model, y_test)))
    return (y_pred_model, history)
    #return y_pred_model

deleteCols = ['Unnamed: 0', 'unit']
clsCol = ['cls']
modeCols = ['mode1', 'mode2', 'mode3', 'mode4', 'mode5', 'mode6']
# chi square selected features
feature1Cols = ['feature1', 'feature2', 'feature5', 'feature6', 'feature8', 'feature10', 'feature11', 'feature13','feature14', 'feature15', 'feature16', 'feature17', 'feature19', 'feature20', 'feature21']
# PCA selected features
feature2Cols = ['feature2','feature3','feature4', 'feature5',  'feature9', 'feature10', 'feature11', 'feature13','feature14', 'feature15', 'feature16', 'feature17', 'feature19', 'feature20', 'feature21']
cycleCol = ['cycle']
settingCols = ['setting1', 'setting2', 'setting3']

# RBF adopted from: https://github.com/PetraVidnerova/rbf_keras
import random
from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import RandomUniform, Initializer, Orthogonal, Constant
import numpy as np


class InitCentersRandom(Initializer):
    """ Initializer for initialization of centers of RBF network
        as random samples from the given data set.
    # Arguments
        X: matrix, dataset to choose the centers from (random rows
          are taken as centers)
    """

    def __init__(self, X):
        self.X = X

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]
        idx = np.random.randint(self.X.shape[0], size=shape[0])
        return self.X[idx, :]


class RBFLayer(Layer):
    """ Layer of Gaussian RBF units.
    # Example

    ```python
        model = Sequential()
        model.add(RBFLayer(10,
                           initializer=InitCentersRandom(X),
                           betas=1.0,
                           input_shape=(1,)))
        model.add(Dense(1))
    ```

    # Arguments
        output_dim: number of hidden units (i.e. number of outputs of the layer)
        initializer: instance of initiliazer to initialize centers
        betas: float, initial value for betas
    """

    def __init__(self, output_dim, initializer=None, betas=1.0, **kwargs):
        self.output_dim = output_dim
        self.init_betas = betas
        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0)
            # self.initializer = Orthogonal()
        else:
            self.initializer = initializer
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim,),
                                     initializer=Constant(value=self.init_betas),
                                     # initializer='ones',
                                     trainable=True)

        super(RBFLayer, self).build(input_shape)

    def call(self, x):

        C = K.expand_dims(self.centers)
        H = K.transpose(C - K.transpose(x))
        return K.exp(-self.betas * K.sum(H ** 2, axis=1))

        # C = self.centers[np.newaxis, :, :]
        # X = x[:, np.newaxis, :]

        # diffnorm = K.sum((C-X)**2, axis=-1)
        # ret = K.exp( - self.betas * diffnorm)
        # return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


y_test0 = None
y_test = None
def executeModel(inputTrainset, inputTestset, columnsToDrop, numOfNodes=100, epochs=100, batch_size = None, verbose=True, isRBF=False):
    print('==============================================================================')
    print ('Delete columns: ')
    print (columnsToDrop)
    print ('Num of ndoes: ' + str(numOfNodes))
    print ('Epochs: ' + str(epochs))
    print('==============================================================================')
    trainset = inputTrainset
    testset = inputTestset
    #Remove columns
    trainset = trainset.drop(columnsToDrop, axis = 1)
    testset = testset.drop(columnsToDrop, axis=1)
    #y_train, X_train
    y_train = trainset['RUL'].values
    X_train = trainset.drop('RUL', axis=1)
    #Converting to array
    X_train = X_train.values

    #y_test, X_test
    y_test0 = testset['RUL']
    y_test = y_test0.values
    X_test = testset.drop('RUL', axis=1)
    #Converting to array
    X_test = X_test.values
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Building the model
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import RMSprop

    model = Sequential()
    layer1 = Dense(units=numOfNodes, activation='relu', input_dim = len(X_train[0]))
    layer2 = Dense(units=1, activation='relu')
    optimizer = 'adam'
    if(isRBF):
        layer1 = RBFLayer(10,
                        initializer=InitCentersRandom(X_train),
                        betas=2.0,
                        input_shape=(len(X_train[0]),))
        layer2 = Dense(units=1)
        optimizer = RMSprop()
    model.add(layer1)
    model.add(layer2)
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    #Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    #Predict
    y_pred_model = model.predict(X_test)
    print('THE MODEL TRAIN SCORE IS: ' +str(history.history['loss'][-1]) + ', TEST SCORE IS: ' +str(calculateScore(y_pred_model, y_test)))
    plotUnitLines(y_pred_model, 1, 20, y_test0)
    return (y_pred_model, history)

print('MODEL: 1 HIDDEN LAYER, 100 NODES, 100 EPOCHS')

#y_pred_mod = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols+feature1Cols, 100, 100, None, False)

# y_pred_mod1, history1 = executeModel(trainset0, testset0, deleteCols+feature1Cols, 100, 100, None, False, False)
# y_pred_mod2, history2 = executeModel(trainset0, testset0, deleteCols+clsCol+feature1Cols, 100, 100, None, False, False)
# y_pred_mod3, history3 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols+feature1Cols, 100, 100, None, False, False)
# y_pred_mod4, history4 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols+cycleCol+feature1Cols, 100, 100, None, False, False)
# y_pred_mod5, history5 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols+cycleCol+settingCols+feature1Cols, 100, 100,None, False, False)
#
# print('MODEL: 1 HIDDEN LAYER, 100 NODES, 200 EPOCHS')
# y_pred_mod6, history6 = executeModel(trainset0, testset0, deleteCols+feature1Cols, 100, 200, None, False, False)
# y_pred_mod7, history7 = executeModel(trainset0, testset0, deleteCols+clsCol+feature1Cols, 100, 200, None, False, False)
# y_pred_mod8, history8 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols+feature1Cols, 100, 200, None, False, False)
# y_pred_mod9, history9 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols+cycleCol+feature1Cols, 100, 200, None, False, False)
# y_pred_mod10, history10= executeModel(trainset0, testset0, deleteCols+clsCol+modeCols+cycleCol+settingCols+feature1Cols, 100, 200,None, False, False)
#
# print('MODEL: 1 HIDDEN LAYER, 100 NODES, 300 EPOCHS')
# y_pred_mod11, history11 = executeModel(trainset0, testset0, deleteCols+feature1Cols, 100, 300, None, False, False)
# y_pred_mod12, history12 = executeModel(trainset0, testset0, deleteCols+clsCol+feature1Cols, 100, 300, None, False, False)
# y_pred_mod13, history13 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols+feature1Cols, 100, 300, None, False, False)
# y_pred_mod14, history14 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols+cycleCol+feature1Cols, 100, 300, None, False, False)
# y_pred_mod15, history15 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols+cycleCol+settingCols+feature1Cols, 100, 300, None, False, False)

#y_pred_mod = executeModel1(trainset1, testset1, deleteCols+clsCol+feature2Cols+modeCols, 100, 100, None, False)

y_pred_mod = executeModel(trainset1, testset1, deleteCols+clsCol+modeCols+feature1Cols, 100, 100, None, False)

# plotLossHistory(history1)
# plotLossHistory(history2)
# plotLossHistory(history3)
# plotLossHistory(history4)
# plotLossHistory(history5)
# plotLossHistory(history6)
# plotLossHistory(history7)
# plotLossHistory(history8)
# plotLossHistory(history9)
# plotLossHistory(history10)
# plotLossHistory(history11)
# plotLossHistory(history12)
# plotLossHistory(history13)
# plotLossHistory(history14)
# plotLossHistory(history15)
