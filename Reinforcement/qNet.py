import gameHandle as gg
import numpy as np
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras import backend as K
import random
import time
import pickle
import copy


GAMMA = 0.9
LEARNING_RATE = 0.0001


MEMORY_SIZE = 40000
BATCH_SIZE = 32

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.9999

def huber_loss(a, b, in_keras=True):
    error = a - b
    quadratic_term = error*error / 2
    linear_term = abs(error) - 1/2
    use_linear_term = (abs(error) > 1.0)
    if in_keras:
        # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
        use_linear_term = K.cast(use_linear_term, 'float32')
    return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term

class DQN:
    def __init__(self, action_space, shape, loadedModel=None, memory=None, history=None):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.shape = shape
        self.tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))

        self.rounds = 0
        self.history = {
            'acc': [],
            'loss': [],
            'qval': [],
            'score': [],
            'moves': []
            }
        huber_loss = self.huber_loss
        if loadedModel==None:
            self.model = Sequential()
            self.model.add(Conv2D(256, 3, strides=1, input_shape=(shape, shape, 1), activation="relu", padding="same"))
            # self.model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding="same"))
            self.model.add(Conv2D(128, 3, strides=1, activation="relu", padding="same"))
            self.model.add(Conv2D(128, 3, strides=1, activation="relu", padding="same"))
            self.model.add(Conv2D(64, 3, strides=1, activation="relu", padding="same"))
            self.model.add(Conv2D(32, 3, strides=1, activation="relu", padding="same"))
            self.model.add(Flatten())
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dense(self.action_space, activation='linear'))
            self.model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE), metrics=["accuracy"])
        else:
            self.memory = memory
            self.history = history
            self.exploration_rate = 0.1
            self.model = loadedModel
        
        self.modelCopy = self.copy_model(self.model)
    
    def copy_model(self, model):
        """Returns a copy of a keras model."""
        model.save('tmp_model')
        return load_model('tmp_model')

        # return load_model('tmp_model', custom_objects={'huber_loss': self.huber_loss})
    
    def huber_loss(self, a, b, in_keras=True):
        error = a - b
        quadratic_term = error * error / 2
        linear_term = abs(error) - 1/2
        use_linear_term = (abs(error) > 1.0)
        if in_keras:
            use_linear_term = K.cast(use_linear_term, 'float32')
        return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term



    def remember(self, state, action, reward, next_state, done):
        self.memory.append((np.array(state).reshape(-1, self.shape, self.shape, 1), action, reward, np.array(next_state).reshape(-1, self.shape, self.shape, 1), done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)

        state = np.array(state).reshape(-1, self.shape, self.shape, 1)
        q_values = self.modelCopy.predict(state)
        return np.argmax(q_values[0])


    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, done in batch:
            q_update = reward
            if not done:
                state_next = np.array(state_next).reshape(-1, self.shape, self.shape, 1)
                q_update = (reward + GAMMA * np.amax(self.modelCopy.predict(state_next)[0]))
                self.history['qval'].append(q_update)
            state = np.array(state).reshape(-1, self.shape, self.shape, 1)
            q_values = self.modelCopy.predict(state)
            q_values[0][action] = q_update
            hist = self.model.fit(state, q_values, verbose=0)
            self.history['loss'].append(hist.history['loss'][0])
            self.rounds += 1
            if self.rounds % 50000 == 0:
                self.modelCopy = self.copy_model(self.model)
                print("ROUNDS {}".format(self.rounds))
                print("DUMPING HISTORY AND MEMORY")
                f = open('history.pckl', 'wb')
                pickle.dump(self.history, f)
                f.close()
                fi = open('memory.pckl', 'wb')
                pickle.dump((self.memory), fi)
                fi.close()

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def trainMaze(loadedModel=None, memory=None, history=None):
    size = 12

    env = gg.Handler(size)
    action_space = 4
    run = 0

    if load_model==None:
        dqn_solver = DQN(action_space, size*2)
    else:
        run = len(history["acc"])
        dqn_solver = DQN(action_space, size*2, loadedModel, memory, history)

    env.render()
    while True:
        run += 1
        state = env.reset()
        step = 0
        terminal = False
        while not terminal:
            action = dqn_solver.act(state)
            state_next, reward, terminal = env.step(action)
            step += reward

            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                dqn_solver.history["acc"].append(env.correctMoves/env.totalMoves)
                dqn_solver.history["score"].append(step)
                dqn_solver.history["moves"].append(env.game.moveCounter)
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step) + ", accuracy: " + str(env.correctMoves/env.totalMoves) +", final accuracy: " + str(env.game.minMoves)+"/"+str(env.totalMoves))

            dqn_solver.experience_replay()

        if run % 30 == 0:
            print(len(dqn_solver.memory))
            dqn_solver.model.save('my_model.h5')

def playGame(games):
    model = load_model('my_model.h5')
    env = gg.Handler(12)
    env.render()

    accuracy = 0


    for _ in range(games):
        state=env.reset()
        terminal = False
        while not terminal:
            time.sleep(0.2)
            move = np.argmax(model.predict(state))
            state, _, terminal = env.step(move)
        accuracy += env.correctMoves/env.totalMoves
        print("Accuracy: " + str(env.correctMoves/env.totalMoves))

    print("Accuracy after {} games = {}".format(games, accuracy/games))



if __name__ == "__main__":
    # f = open('memory.pckl', 'rb')
    # memory = (pickle.load(f))
    # f.close()
    # f1 = open('history.pckl', 'rb')
    # history = pickle.load(f1)
    # f1.close()
    # model = load_model('my_model.h5')
    # model.summary()

    # trainMaze(model, memory, history)
    
    # trainMaze()
    playGame(100)
