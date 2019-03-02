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
            self.game.mazeLevelOne(self.shape)
            if fg.isValidMap(self.game.map): break

        if self.graphics:
            self.graphicsHandler.map = self.game.map
            self.graphicsUpdate()
        return self.game.map.grid

    def render(self):
        self.graphics = True
        self.graphicsHandler = fg.Graphics(25, self.game.map)
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
        self.game.makeMove(move)

        if self.graphics: self.graphicsUpdate()

        postMove = fg.closestMoves(self.game.map)
        postMoveObjectives = self.game.objectiveCounter

        if len(postMove) == 0:
            return np.array(self.game.map.grid).reshape(-1, self.shape, self.shape, 1), 10, True

        if (self.game.moveCounter > 200):
            return np.array(self.game.map.grid).reshape(-1, self.shape, self.shape, 1), -10, True

        if (len(preMove) < len(postMove)):
            return np.array(self.game.map.grid).reshape(-1, self.shape, self.shape, 1), len(postMove), False

        if (postMoveObjectives > preMoveObjectives):
            return np.array(self.game.map.grid).reshape(-1, self.shape, self.shape, 1), 5, False

        if len(preMove) > len(postMove):
            return np.array(self.game.map.grid).reshape(-1, self.shape, self.shape, 1), -len(postMove), False

        if len(preMove) == len(postMove):
            return np.array(self.game.map.grid).reshape(-1, self.shape, self.shape, 1), -1, False

#
# example = Handler()
# example.reset()
# example.render()
# games = 0
# while 1:
#     try:
#         move = fg.closestMoves(example.game.map)[0]
#     except Exception:
#         example.render()
#         time.sleep(5)
#         break
#     _, _, done = example.step(move)
#     if done:
#         games += 1
#         example.reset()
