import tensorflow as tf
import numpy as np
import json
import copy
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.optimizers import Adam
from keras import losses
import pickle
import gameHandle as gg

INPUT_SIZE = 24
LEARNING_RATE = 0.0001
BATCH_SIZE = 512
TRAINING_ITERS = 101



class SupervisedNet:
    def __init__(self, action_space):
        self.action_space = action_space
        
        self.model = Sequential()
        self.model.add(Conv2D(32, 3, strides=1, input_shape=(INPUT_SIZE, INPUT_SIZE, 1), activation="relu", padding="same"))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding="same"))
        self.model.add(Conv2D(64, 3, strides=1, activation="relu", padding="same"))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding="same"))
        self.model.add(Conv2D(128, 3, strides=1, activation="relu", padding="same"))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding="same"))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation="relu"))
        self.model.add(Dense(self.action_space, activation='softmax'))
        self.model.compile(loss=losses.categorical_crossentropy, optimizer=Adam(lr=LEARNING_RATE), metrics=["accuracy"])

        
def loadMaps():
    with open('maps1.json', 'r') as f:
        a = json.load(f)
    return copy.deepcopy(a)

# SHAPE OF THE DATA
# print(np.array(maps["maps"]).shape)
# # SHAPE OF THE LABELS
# print(np.array(maps["moves"]).shape)

def prepareData():
    maps = loadMaps()
    npSolution = np.array(maps["moves"]).reshape(len(maps["moves"]), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    encoded = onehot_encoder.fit_transform(npSolution)
    X_train, X_test, y_train, y_test = train_test_split(np.array(maps["maps"]), encoded, test_size=0.3, random_state = 101)
    train_X = X_train.reshape(-1, INPUT_SIZE, INPUT_SIZE, 1)
    test_X = X_test.reshape(-1, INPUT_SIZE, INPUT_SIZE, 1)

    return train_X, y_train, test_X, y_test



def playGame(games):
    model = load_model('models/12x12.h5')
    game = gg.Handler(12)
    game.render()
    
    accuracy = 0

    for _ in range(games):
        terminal = False
        state = game.reset()
        while not terminal:
            move = np.argmax(model.predict(state))
            state, _, terminal = game.step(move)
        accuracy += game.correctMoves/game.totalMoves
        print("Accuracy: " + str(game.correctMoves/game.totalMoves))
    print("Accuracy after {} games = {}".format(games, accuracy/games))




def trainModel():
    network = SupervisedNet(4)
    train_X, y_train, test_X, y_test = prepareData()

    trainHistory = {
        'loss': [],
        'accuracy': []
    }

    testHistory = {
        'loss': [],
        'accuracy': []
    }

    for i in range(TRAINING_ITERS):
        for batch in range(len(train_X)//BATCH_SIZE):
            batch_x = train_X[batch*BATCH_SIZE:min((batch+1)*BATCH_SIZE,len(train_X))]
            batch_y = y_train[batch*BATCH_SIZE:min((batch+1)*BATCH_SIZE,len(y_train))] 

            history = network.model.fit(batch_x, batch_y, verbose=False) 

            trainHistory['loss'].append(history.history['loss'])
            trainHistory['accuracy'].append(history.history['acc'])

        evaluation = network.model.evaluate(test_X, y_test, verbose=False)
        testHistory['loss'].append(evaluation[0])
        testHistory['accuracy'].append(evaluation[1])

        if (i % 10 == 0):
            print("ITERATION: {}".format(i))
            f = open('train/12x12.pckl', 'wb')
            pickle.dump(trainHistory, f)
            f.close()
            f1 = open('test/12x12.pckl', 'wb')
            pickle.dump(testHistory, f1)
            f1.close()
    network.model.save('models/12x12.h5')


if __name__ == "__main__":
    # trainModel()
    playGame(1000)



