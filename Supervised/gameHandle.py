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
            self.game.testMaze(self.shape, 11)
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
# game = Handler(12)
# game.render()

# predictionTime = 0
# games = 0

# while 1:
#     game.reset()
#     games += 1
#     before = time.time()
#     moves = fg.closestMoves(game.game.map)
#     predictionTime += time.time() - before
#     while(len(moves) > 0):
#         _, done = game.longMove(moves.pop(0))
#     print("Avg Time Taken after {} games: {}".format(games, predictionTime/games))

