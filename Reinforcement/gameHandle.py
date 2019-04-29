import finalGame as fg
import random
import cv2
import numpy as np
import time


class Handler():
    def __init__(self, shape, mapType):
        self.game = None
        self.graphics = False
        self.shape = shape
        self.paddedMap = None
        self.moveEpisode = 0
        self.totalMoves = 0
        self.correctMoves = 0
        self.reset(mapType)

    def reset(self, mapType):
        self.moveEpisode = 0
        self.totalMoves = 0
        self.correctMoves = 0
        while 1:
            self.game = fg.Game()
            if mapType == 0:
                self.game.testMaze(12, 1)
                equalMap = False
                for i in self.game.testMaps["short"]:
                    if np.array_equal(np.array(i).reshape(12,12), self.game.map.grid):
                        equalMap = True
                        print("REPEATED")
                        break
                if equalMap:
                    continue
            if mapType == 1:
                self.game.testMaze(12, 11)
                equalMap = False
                for i in self.game.testMaps["tsp"]:
                    if np.array_equal(np.array(i).reshape(12,12), self.game.map.grid):
                        equalMap = True
                        break
                if equalMap:
                    continue
            if mapType == 2:
                self.game.createGrid(12, 7)
                equalMap = False
                for i in self.game.testMaps["saltpepper"]:
                    if np.array_equal(np.array(i).reshape(12,12), self.game.map.grid):
                        equalMap = True
                        break
                if equalMap:
                    continue
            if mapType == 3:
                self.game.mazeLevelOne(12, 7)
                equalMap = False
                for i in self.game.testMaps["maze"]:
                    if np.array_equal(np.array(i).reshape(12,12), self.game.map.grid):
                        equalMap = True
                        break
                if equalMap:
                    continue
            
            self.paddedMap = self.game.map.getCentricPosition()
            if fg.isValidMap(self.game.map):
                break

        if self.graphics:
            self.graphicsHandler.map = self.paddedMap
            self.graphicsUpdate()
        return np.array(self.paddedMap.grid).reshape(-1, len(self.paddedMap.grid), len(self.paddedMap.grid), 1)

    def longMove(self, move):
        self.game.makeMove(move)
        self.paddedMap = self.game.map.getCentricPosition()
        if self.graphics:
            self.graphicsUpdate()

        self.totalMoves += 1
        if (self.game.objectiveCounter == self.game.map.maxObjectives):
            return np.array(self.paddedMap.grid).reshape(-1, len(self.paddedMap.grid), len(self.paddedMap.grid), 1), True
        else:
            return np.array(self.paddedMap.grid).reshape(-1, len(self.paddedMap.grid), len(self.paddedMap.grid), 1), False    

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
# game = Handler(12, 0)
# game.render()
# moves = fg.closestMoves(game.game.map)

# moveCounter = 0
# while 1:
#     time.sleep(0.1)
#     moves = fg.closestMoves(game.game.map)
#     if len(moves) == 0:
#         game.reset(0)
#         continue
#     _, reward, done = game.step(moves.pop(0))
#     moveCounter += 1
#     if done:
#         print("TSP SOLVER: {}, FINAL SOLVE: {}".format(game.game.minMoves, moveCounter))
#         moveCounter = 0
#         game.reset(0)
#         continue
