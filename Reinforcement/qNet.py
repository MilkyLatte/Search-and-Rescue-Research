import gameHandle as gg
import numpy as np
import test
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
EXPLORATION_MIN = 0.05
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
            print("HERE")
            self.memory = memory
            self.history = history
            self.exploration_rate = 0.05
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

    env = gg.Handler(size, 2)
    action_space = 4
    run = 0

    if load_model is  None:
        dqn_solver = DQN(action_space, size*2)
    else:
        run = len(history["acc"])
        dqn_solver = DQN(action_space, size*2, loadedModel, memory, history)

    env.render()
    while True:
        run += 1
        state = env.reset(2)
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


def testGame():
    model = load_model('./finalModels/models/TSP.h5')
    env = test.Test(12, 1)
    env.render()
    accuracy = 0
    moves = 0
    avgtime = 0
    avgtimePerGame = 0
    games = 0
    variance = 0
    accMean = []

    for _ in range(len(env.game.testMaps["maze"]) - 1):
        state=env.nextGame(1)
        terminal = False
        before = time.time()

        while not terminal:
            # time.sleep(0.1)
            pre = time.time()
            move = np.argmax(model.predict(state))
            if np.random.rand() < 0.05:
                move = np.random.randint(0, 4)
            avgtime += abs(time.time() - pre)
            moves += 1
            state, _, terminal = env.step(move)
        accuracy += env.correctMoves/env.totalMoves
        accMean.append(env.correctMoves/env.totalMoves)
        games += 1
        avgtimePerGame += abs(time.time() - before)
        variance += abs(env.game.minMoves - env.totalMoves)
    print("Time Average per move: {} seconds".format(avgtime/moves))
    print("Time Average per game: {} seconds".format(avgtimePerGame/games))
    print("Accuracy after {} games = {}, Average Move Variance = {} ".format(games, accuracy/(games), variance/games))
    print("AVERAGE ACCURACY: {}".format(np.mean(np.array(accMean))))
    print("VARIANCE: {}".format(np.var(np.array(accMean))))


def playGame(games):
    model = load_model('./finalModels/models/TSP.h5')
    model.summary()
    game = gg.Handler(12, 1)
    game.render()
    

    timeTaken = 0
    variance = 0
    accuracy = 0
    accMean = []
    

    for _ in range(games):
        terminal = False
        state = game.reset(1)
        before = time.time()
        while not terminal:
            # time.sleep(1)
            move = np.argmax(model.predict(state))
            state,  terminal = game.longMove(move)
        timeTaken += time.time() - before
        variance += game.totalMoves - game.game.minMoves
        accuracy += game.correctMoves/game.totalMoves 
        accMean.append(game.correctMoves/game.totalMoves)
    print("AVERAGE TIME TAKEN: {}".format(timeTaken/games))
    print("AVERAGE VARIANCE: {}".format(variance/games))
    print("AVERAGE ACCURACY: {}".format(np.mean(np.array(accMean))))
    print("VARIANCE: {}".format(np.var(np.array(accMean))))

if __name__ == "__main__":
    # f = open('./finalModels/memory/SaltPepperMem.pckl', 'rb')
    # memory = (pickle.load(f))
    # f.close()
    # f1 = open('./finalModels/history/SaltPepperHist.pckl', 'rb')
    # history = pickle.load(f1)
    # f1.close()
    # model = load_model('./finalModels/models/SaltPepperModel.h5')
    # model.summary()

    # trainMaze(model, memory, history)
    
    # trainMaze()
    testGame()
    # playGame(10)
