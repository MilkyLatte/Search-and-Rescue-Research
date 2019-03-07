import finalGame as fg
import random
import cv2
import numpy as np
import time


class Handler():
    def __init__(self, shape):
        self.game = None
        self.graphics = False
        self.shape = shape
        self.reset()

    def reset(self):
        while 1:
            self.game = fg.Game()
            self.game.testMaze(self.shape, 2)
            if fg.isValidMap(self.game.map):
                break

        if self.graphics:
            self.graphicsHandler.map = self.game.map
            self.graphicsUpdate()
        return self.game.map.grid

    def render(self):
        self.graphics = True
        self.graphicsHandler = fg.Graphics(50, self.game.map)
        self.graphicsHandler.createBoard()
        self.graphicsHandler.board.pack()
        self.graphicsHandler.master.update()

    def graphicsUpdate(self):
        self.graphicsHandler.updateBoard()
        self.graphicsHandler.board.pack()
        self.graphicsHandler.master.update()

    def step(self, move):
        preMove = fg.closestMoves(self.game.map)
        preMoveObjectives = self.game.objectiveCounter
        previous = self.game.makeMove(move)

        if self.graphics: self.graphicsUpdate()

        postMove = fg.closestMoves(self.game.map)
        postMoveObjectives = self.game.objectiveCounter

        accumulator = 0
        if previous: accumulator = -0.1


        if len(postMove) == 0:
            return np.array(self.game.map.grid).reshape(-1, self.shape, self.shape, 1), 1, True

        if (self.game.moveCounter > 500):
            return np.array(self.game.map.grid).reshape(-1, self.shape, self.shape, 1), -1, True

        if len(preMove) == len(postMove):
            return np.array(self.game.map.grid).reshape(-1, self.shape, self.shape, 1), -1, False

        if (postMoveObjectives > preMoveObjectives):
            return np.array(self.game.map.grid).reshape(-1, self.shape, self.shape, 1), 1, False

        if (len(preMove) < len(postMove)):
            return np.array(self.game.map.grid).reshape(-1, self.shape, self.shape, 1), -1/len(preMove) + accumulator, False

        if len(preMove) > len(postMove):
            return np.array(self.game.map.grid).reshape(-1, self.shape, self.shape, 1), 1/len(postMove), False


# done = False
# game = Handler(20)
# game.render()
# moves = fg.closestMoves(game.game.map)
#
#
# while not done:
#     moves = fg.closestMoves(game.game.map)
#     time.sleep(0.2)
#     _, reward, done = game.step(moves.pop(0))
#     print("REWARD: {}".format(reward))
