import gameHandle as gg
import numpy as np
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense, Convolution2D, Flatten
from keras.optimizers import Adam
import random
import time

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 20000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.0001
EXPLORATION_DECAY = 0.9999


class DQN:
    def __init__(self, action_space, shape):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.shape = shape

        self.model = Sequential()
        self.model.add(Convolution2D(32, 8, 8, subsample=(4,4), input_shape=(shape, shape, 1), activation="relu"))
        self.model.add(Convolution2D(64, 4, 4, subsample=(4,4), activation="relu"))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

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
            state = np.array(state).reshape(-1, self.shape, self.shape, 1)
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def trainMaze():
    env = gg.Handler(20)

    action_space = 4
    dqn_solver = DQN(action_space, 20)

    env.render()
    run = 0
    while True:
        run += 1
        state = env.reset()
        step = 0
        terminal = False
        while not terminal:
            step += 1
            action = dqn_solver.act(state)
            state_next, reward, terminal = env.step(action)

            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print ("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))

            dqn_solver.experience_replay()
        if run % 30 == 0:
            print(len(dqn_solver.memory))
            dqn_solver.model.save('my_model.h5')


if __name__ == "__main__":
    trainMaze()
