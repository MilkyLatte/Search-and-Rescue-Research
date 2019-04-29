import finalGame as fg
import random
import cv2
import numpy as np
import time
import json


class Test():
    def __init__(self, shape, mapType):
        self.game = fg.Game()
        self.graphics = False
        self.shape = shape
        self.paddedMap = None
        self.moveEpisode = 0
        self.totalMoves = 0
        self.correctMoves = 0
        self.gameCounter = 0
        self.nextGame(mapType)

    def nextGame(self, mapType):
        self.moveEpisode = 0
        self.totalMoves = 0
        self.correctMoves = 0
        if (mapType == 0):
            self.game.loadGrid(np.array(self.game.testMaps["short"][self.gameCounter]).reshape(12,12))
        if (mapType == 1):
            self.game.loadGrid(np.array(self.game.testMaps["tsp"][self.gameCounter]).reshape(12,12))
        if (mapType == 2):
            self.game.loadGrid(np.array(self.game.testMaps["saltpepper"][self.gameCounter]).reshape(12,12))
        if (mapType == 3):
            self.game.loadGrid(np.array(self.game.testMaps["maze"][self.gameCounter]).reshape(12,12))
        self.paddedMap = self.game.map.getCentricPosition()
        self.gameCounter += 1

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
            _, preMove = fg.brute_force(self.game.map)
            preMoveObjectives = self.game.objectiveCounter
            previous = self.game.makeMove(move)
            self.moveEpisode += 1
            self.totalMoves += 1


            self.paddedMap = self.game.map.getCentricPosition()

            if self.graphics:
                self.graphicsUpdate()

            _, postMove = fg.brute_force(self.game.map)
            postMoveObjectives = self.game.objectiveCounter


            if (postMove) == 0:
                self.correctMoves += 1
                return np.array(self.paddedMap.grid).reshape(-1, len(self.paddedMap.grid), len(self.paddedMap.grid), 1), 1, True

            if (self.moveEpisode > 70):
                return np.array(self.paddedMap.grid).reshape(-1, len(self.paddedMap.grid), len(self.paddedMap.grid), 1), 0, True

            if (preMove) == (postMove):
                return np.array(self.paddedMap.grid).reshape(-1, len(self.paddedMap.grid), len(self.paddedMap.grid), 1), -1, False

            if (postMoveObjectives > preMoveObjectives):
                self.correctMoves += 1
                self.moveEpisode = 0
                return np.array(self.paddedMap.grid).reshape(-1, len(self.paddedMap.grid), len(self.paddedMap.grid), 1), 1, False

            if (preMove) > (postMove):
                self.correctMoves += 1
                return np.array(self.paddedMap.grid).reshape(-1, len(self.paddedMap.grid), len(self.paddedMap.grid), 1), 1/(postMove), False

            if previous:
                return np.array(self.paddedMap.grid).reshape(-1, len(self.paddedMap.grid), len(self.paddedMap.grid), 1), -0.5, False

            if ((preMove) < (postMove)):
                return np.array(self.paddedMap.grid).reshape(-1, len(self.paddedMap.grid), len(self.paddedMap.grid), 1), -1/(preMove), False

# done = False
# game = Test(12, 3)
# game.render()
# moves = fg.closestMoves(game.game.map)


# while 1:
#     time.sleep(0.1)
#     moves = fg.closestMoves(game.game.map)
#     if len(moves) == 0:
#         game.nextGame(3)
#         continue
#     _, reward, done = game.step(moves.pop(0))
#     if done:
#         print("TSP SOLVER: {}, FINAL SOLVE: {}".format(game.game.minMoves, game.totalMoves))
#         game.nextGame(3)
#         continue