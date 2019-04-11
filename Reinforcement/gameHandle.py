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
        self.paddedMap = None
        self.moveEpisode = 0
        self.totalMoves = 0
        self.correctMoves = 0
        self.reset()

    def reset(self):
        self.moveEpisode = 0
        self.totalMoves = 0
        self.correctMoves = 0
        while 1:
            self.game = fg.Game()
            self.game.mazeLevelOne(self.shape, 7)
            self.paddedMap = self.game.map.getCentricPosition()
            if fg.isValidMap(self.game.map):
                break

        if self.graphics:
            self.graphicsHandler.map = self.paddedMap
            self.graphicsUpdate()
        return np.array(self.paddedMap.grid).reshape(-1, len(self.paddedMap.grid), len(self.paddedMap.grid), 1)

    def render(self):
        self.graphics = True
        self.graphicsHandler = fg.Graphics(35, self.paddedMap)
        self.graphicsHandler.createBoard()
        self.graphicsHandler.board.pack()
        self.graphicsHandler.master.update()

    def graphicsUpdate(self):
        self.graphicsHandler.map = self.paddedMap
        self.graphicsHandler.updateBoard()
        self.graphicsHandler.board.pack()
        self.graphicsHandler.master.update()

    def step(self, move):
        preMove = fg.closestMoves(self.game.map)
        preMoveObjectives = self.game.objectiveCounter
        previous = self.game.makeMove(move)
        self.moveEpisode += 1
        self.totalMoves += 1


        self.paddedMap = self.game.map.getCentricPosition()

        if self.graphics:
            self.graphicsUpdate()

        postMove = fg.closestMoves(self.game.map)
        postMoveObjectives = self.game.objectiveCounter


        if len(postMove) == 0:
            self.correctMoves += 1
            return np.array(self.paddedMap.grid).reshape(-1, len(self.paddedMap.grid), len(self.paddedMap.grid), 1), 1, True

        if (self.moveEpisode > 70):
            return np.array(self.paddedMap.grid).reshape(-1, len(self.paddedMap.grid), len(self.paddedMap.grid), 1), 0, True

        if len(preMove) == len(postMove):
            return np.array(self.paddedMap.grid).reshape(-1, len(self.paddedMap.grid), len(self.paddedMap.grid), 1), -1, False

        if (postMoveObjectives > preMoveObjectives):
            self.correctMoves += 1
            self.moveEpisode = 0
            return np.array(self.paddedMap.grid).reshape(-1, len(self.paddedMap.grid), len(self.paddedMap.grid), 1), 1, False

        if len(preMove) > len(postMove):
            self.correctMoves += 1
            return np.array(self.paddedMap.grid).reshape(-1, len(self.paddedMap.grid), len(self.paddedMap.grid), 1), 1/len(postMove), False

        if previous:
            return np.array(self.paddedMap.grid).reshape(-1, len(self.paddedMap.grid), len(self.paddedMap.grid), 1), -0.5, False

        if (len(preMove) < len(postMove)):
            return np.array(self.paddedMap.grid).reshape(-1, len(self.paddedMap.grid), len(self.paddedMap.grid), 1), -1/len(preMove), False


# done = False
# game = Handler(12)
# game.render()
# moves = fg.closestMoves(game.game.map)


# while 1:
#     moves = fg.closestMoves(game.game.map)
#     if len(moves) == 0:
#         game.reset()
#         continue
#     _, reward, done = game.step(moves.pop(0))
#     if done:
#         print("TSP SOLVER: {}, FINAL SOLVE: {}".format(game.game.minMoves, game.totalMoves))
#         game.reset()
#         continue
