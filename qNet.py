import gameHandle as gg
import numpy as np
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import random
import time
import pickle

GAMMA = 0.9
LEARNING_RATE = 0.0001

MEMORY_SIZE = 12000
BATCH_SIZE = 32

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_DECAY = 0.9995


class DQN:
    def __init__(self, action_space, shape, loadedModel=None, memory=None):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)
        print(type(self.memory))

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
        if loadedModel==None:
            self.model = Sequential()
            self.model.add(Conv2D(64, 3, strides=2, input_shape=(shape, shape, 1), activation="relu", padding="same"))
            self.model.add(Conv2D(32, 4, strides=2, activation="relu", padding="same"))
            self.model.add(Conv2D(16, 8, strides=2, activation="relu", padding="same"))
            self.model.add(Flatten())
            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dense(self.action_space))
            self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE), metrics=["accuracy"])
        else:
            self.memory = memory
            self.exploration_rate = 0.1
            self.model = loadedModel


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)

        state = np.array(state).reshape(-1, self.shape, self.shape, 1)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])


    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, done in batch:
            q_update = reward
            if not done:
                state_next = np.array(state_next).reshape(-1, self.shape, self.shape, 1)
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
                self.history['qval'].append(q_update)
            state = np.array(state).reshape(-1, self.shape, self.shape, 1)
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            hist = self.model.fit(state, q_values, verbose=0)
            self.history['acc'].append(hist.history['acc'][0])
            self.history['loss'].append(hist.history['loss'][0])
            self.rounds += 1
            if self.rounds % 50000 == 0:
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


def trainMaze(loadedModel=None, memory=None):

    env = gg.Handler(12)
    action_space = 4

    if load_model==None:
        dqn_solver = DQN(action_space, 12)
    else:
        dqn_solver = DQN(action_space, 12, loadedModel, memory)

    env.render()
    run = 0
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
                dqn_solver.history["score"].append(step)
                dqn_solver.history["moves"].append(env.game.moveCounter)
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step) + ", moves: " + str(env.game.moveCounter))

            dqn_solver.experience_replay()

        if run % 30 == 0:
            print(len(dqn_solver.memory))
            dqn_solver.model.save('my_model.h5')


if __name__ == "__main__":
    f = open('models/memory/stage1Mem.pckl', 'rb')
    memory = (pickle.load(f))
    f.close()
    model = load_model('models/stage1Model.h5')
    trainMaze(model, memory)
    # trainMaze()
