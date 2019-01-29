import numpy as np
import tkinter as tk
import time
import pandas as pd
import copy


def smoothMap(grid, x):
    gridCopy = grid
    for i in range(x):
        for j in range(x):
            if(grid[i][j]==2):
                diagonals = 0
                axis = 0
                if grid[(i+1)%x][j]==0:
                    axis += 1
                if grid[(i-1)%x][j]==0:
                    axis+=1
                if grid[i][(j+1)%x]==0:
                    axis+=1
                if grid[i][(j-1)%x]==0:
                    axis+=1
                if grid[(i+1)%x][(j+1)%x]==0:
                    diagonals+=1
                if grid[(i+1)%x][(j-1)%x]==0:
                    diagonals+=1
                if grid[(i-1)%x][(j+1)%x]==0:
                    diagonals+=1
                if grid[(i-1)%x][(j-1)%x]==0:
                    diagonals+=1
                if diagonals == 1 or axis == 4 or axis == 2:
                    gridCopy[i][j] = 0
            if(grid[i][j]==0):
                diagonals = 0
                axis = 0
                if grid[(i+1)%x][j]==2:
                    axis += 1
                if grid[(i-1)%x][j]==2:
                    axis+=1
                if grid[i][(j+1)%x]==2:
                    axis+=1
                if grid[i][(j-1)%x]==2:
                    axis+=1
                if grid[(i+1)%x][(j+1)%x]==2:
                    diagonals+=1
                if grid[(i+1)%x][(j-1)%x]==2:
                    diagonals+=1
                if grid[(i-1)%x][(j+1)%x]==2:
                    diagonals+=1
                if grid[(i-1)%x][(j-1)%x]==2:
                    diagonals+=1
                if diagonals == 1 or axis == 4:
                    gridCopy[i][j] = 2
    return grid

class Game():
    def __init__(self):
        self.moveCounter = 0
        self.objectiveCounter = 0
        self.map =  []


    def createGrid(self, gridSize):
        apX, apY = np.random.randint(0, gridSize), np.random.randint(0, gridSize)
        opX, opY = np.random.randint(0, gridSize), np.random.randint(0, gridSize)
        grid =[]
        grid = [[0 for row in range(gridSize)] for col in range(gridSize)]
        grid[apX][apY] = 1
        grid[opX][opY] = 4
        maxObjectives = 0
        for i in range(gridSize):
            for j in range(gridSize):
                if (i != apX and j != apY and i != opX and j != opY):
                    randomMap = np.random.randint(0, 2)
                    if randomMap == 1:
                        grid[i][j] = 2
        for i in range(15):
            grid = smoothMap(grid, gridSize)
        for i in range(gridSize):
            for j in range(gridSize):
                if (i != apX and j != apY and grid[i][j] != 2):
                    objective = np.random.randint(0, 30)
                    if objective == 1:
                        grid[i][j] = 3
                        maxObjectives += 1
        self.map = Grid(apX, apY, opX, opY, grid, maxObjectives, gridSize)

    def loadGrid(self, grid):
        apX, apY, opX, opY, maxObjectives = 0, 0, 0, 0, 0
        for row in range(len(grid)):
            for col in range(len(grid)):
                if grid[row][col] == 1:
                    apX = row
                    apY = col
                elif grid[row][col] == 4:
                    opX = row
                    opY = col
                elif grid[col][row] == 3:
                    maxObjectives += 1
        self.map = Grid(apX, apY, opX, opY, grid, maxObjectives, len(grid))

    def makeMove(self, move):
        if move == 0:
            pass
        elif move == 1:
            if self.map.grid[(self.map.apX-1)%self.map.size][self.map.apY] != 2:
                self.map.grid[self.map.apX][self.map.apY] = 0
                if self.map.grid[(self.map.apX-1)%self.map.size][self.map.apY] == 3:
                    self.objectiveCounter += 1
                self.map.apX = (self.map.apX - 1)%self.map.size
                self.map.grid[self.map.apX][self.map.apY] = 1
                self.moveCounter += 1
        elif move == 2:
            if self.map.grid[self.map.apX][(self.map.apY-1)%self.map.size] != 2:
                self.map.grid[self.map.apX][self.map.apY] = 0
                if self.map.grid[self.map.apX][(self.map.apY-1)%self.map.size] == 3:
                    self.objectiveCounter += 1
                self.map.apY = (self.map.apY - 1)%self.map.size
                self.map.grid[self.map.apX][self.map.apY] = 1
                self.moveCounter += 1
        elif move == 3:
            if self.map.grid[(self.map.apX+1)%self.map.size][self.map.apY] != 2:
                self.map.grid[self.map.apX][self.map.apY] = 0
                if self.map.grid[(self.map.apX+1)%self.map.size][self.map.apY] == 3:
                    self.objectiveCounter += 1
                self.map.apX = (self.map.apX + 1)%self.map.size
                self.map.grid[self.map.apX][self.map.apY] = 1
                self.moveCounter += 1
        elif move == 4:
            if self.map.grid[self.map.apX][(self.map.apY+1)%self.map.size] != 2:
                self.map.grid[self.map.apX][self.map.apY] = 0
                if self.map.grid[self.map.apX][(self.map.apY+1)%self.map.size] == 3:
                    self.objectiveCounter += 1
                self.map.apY = (self.map.apY + 1)%self.map.size
                self.map.grid[self.map.apX][self.map.apY] = 1
                self.moveCounter += 1


    def getGameState(self):
        return self.map.grid, self.moveCounter, self.objectiveCounter






class Grid():
    def __init__(self, apX, apY, opX, opY, grid, maxObjectives, size):
        self.size = size
        self.grid = grid
        self.apX = apX
        self.apY = apY
        self.opX = opX
        self.opY = opY
        self.maxObjectives = maxObjectives
        self.fullPath = []

    def getObjectives(self):
        objectives = []
        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row][col] == 3:
                    objectives.append((row, col))
        objectives.append((self.opX, self.opY))
        return objectives


class Graphics():
    def __init__(self, width, grid):
        self.master = tk.Tk()
        self.width = width
        self.map = grid

    def createBoard(self):
        self.gridBoard = [[0 for row in range(self.map.size)] for col in range(self.map.size)]
        self.board = tk.Canvas(self.master, width=self.map.size*self.width, height=self.map.size*self.width)
        for i in range(self.map.size):
            for j in range(self.map.size):
                if self.map.grid[i][j] == 0:
                    self.gridBoard[i][j] = self.board.create_rectangle(i*self.width, j*self.width, (i+1)*self.width, (j+1)*self.width, fill="white", width=1)
                elif self.map.grid[i][j] == 1:
                    self.gridBoard[i][j] = self.board.create_rectangle(i*self.width, j*self.width, (i+1)*self.width, (j+1)*self.width, fill="green", width=1)
                elif self.map.grid[i][j] == 2:
                    self.gridBoard[i][j] = self.board.create_rectangle(i*self.width, j*self.width, (i+1)*self.width, (j+1)*self.width, fill="black", width=1)
                elif self.map.grid[i][j] == 3:
                    self.gridBoard[i][j] = self.board.create_rectangle(i*self.width, j*self.width, (i+1)*self.width, (j+1)*self.width, fill="yellow", width=1)
                elif self.map.grid[i][j] == 4:
                    self.gridBoard[i][j] = self.board.create_rectangle(i*self.width, j*self.width, (i+1)*self.width, (j+1)*self.width, fill="purple", width=1)
        for i in range(self.map.size):
            for j in range(self.map.size):
                if self.map.grid[i][j] == 0:
                    self.gridBoard[i][j] = self.board.create_rectangle(i*self.width, j*self.width, (i+1)*self.width, (j+1)*self.width, fill="white", width=1)
                elif self.map.grid[i][j] == 1:
                    self.gridBoard[i][j] = self.board.create_rectangle(i*self.width, j*self.width, (i+1)*self.width, (j+1)*self.width, fill="green", width=1)
                elif self.map.grid[i][j] == 2:
                    self.gridBoard[i][j] = self.board.create_rectangle(i*self.width, j*self.width, (i+1)*self.width, (j+1)*self.width, fill="black", width=1)
                elif self.map.grid[i][j] == 3:
                    self.gridBoard[i][j] = self.board.create_rectangle(i*self.width, j*self.width, (i+1)*self.width, (j+1)*self.width, fill="yellow", width=1)
                elif self.map.grid[i][j] == 4:
                    self.gridBoard[i][j] = self.board.create_rectangle(i*self.width, j*self.width, (i+1)*self.width, (j+1)*self.width, fill="purple", width=1)
    def updateBoard(self):
        if self.map.apX != self.map.opX and self.map.apY != self.map.opY:
            self.map.grid[self.map.opX][self.map.opY] = 4
        for i in range(self.map.size):
            for j in range(self.map.size):
                if self.map.grid[i][j] == 0:
                    self.board.itemconfig(self.gridBoard[i][j], fill="white")
                elif self.map.grid[i][j] == 1:
                    self.board.itemconfig(self.gridBoard[i][j], fill="green")
                elif self.map.grid[i][j] == 4:
                    self.board.itemconfig(self.gridBoard[i][j], fill="purple")


class Node:
    def __init__(self, row, col, dist, parentX, parentY):
        self.row = row
        self.col = col
        self.dist = dist
        self.parentX = parentX
        self.parentY = parentY


class Queue:
  def __init__(self):
      self.queue = list()

  def enqueue(self,data):
      if data not in self.queue:
          self.queue.insert(0,data)
          return True
      return False

  def dequeue(self):
      if len(self.queue)>0:
          return self.queue.pop()
      return ("Queue Empty!")

  def size(self):
      return len(self.queue)

  def printQueue(self):
      return self.queue

  def empty(self):
      if len(self.queue) == 0:
          return True
      return False

class Stack:
    def __init__(self):
        self.stack = list()
    def push(self, data):
        if data not in self.stack:
            self.stack.insert(0, data)
            return True
        return False

    def pop(self):
        return self.stack.pop(0)


def minDist(grid, sourceX, sourceY, targetX, targetY, x):
    source = Node(sourceX, sourceY, 0, -1, -1)
    visited = [[False for row in range(x)] for col in range(x)]
    for row in range(x):
        for col in range(x):
            if grid[row][col] == 2:
                visited[row][col] = True
    s = Stack()
    q = Queue()
    q.enqueue(source)
    s.push(source)
    visited[source.row][source.col] = True
    while(not q.empty()):
        p = q.dequeue()

        if (p.row == targetX and p.col == targetY):
            return p.dist, s

        if visited[(p.row-1)%x][p.col] == False:
            up = Node((p.row-1)%x, p.col, p.dist + 1, p.row, p.col)
            q.enqueue(up)
            s.push(up)
            visited[(p.row-1)%x][p.col] = True

        if visited[(p.row+1)%x][p.col] == False:
            down = Node((p.row+1)%x, p.col, p.dist + 1, p.row, p.col)
            q.enqueue(down)
            s.push(down)
            visited[(p.row+1)%x][p.col] = True

        if visited[p.row][(p.col-1)%x] == False:
            left = Node(p.row, (p.col-1)%x, p.dist + 1, p.row, p.col)
            q.enqueue(left)
            s.push(left)
            visited[p.row][(p.col-1)%x] = True

        if visited[p.row][(p.col+1)%x] == False:
            right = Node(p.row, (p.col+1)%x, p.dist + 1, p.row, p.col)
            q.enqueue(right)
            s.push(right)
            visited[p.row][(p.col+1)%x] = True
    return -1, -1

def getPath(stack, obX, obY, x):
    path = []
    currentPosition = stack.pop()
    while(1):
        currentPosition = stack.pop()
        if currentPosition.row == obX and currentPosition.col == obY:
            break

    path.append(currentPosition)

    for i in range(len(stack.stack)):
        e = stack.pop()
        if e.row == currentPosition.parentX and e.col == currentPosition.parentY:
            currentPosition = e
            path.append(e)
    path.reverse()
    instructions = []
    for i in range(len(path)-1):
        if path[i].row == obX and path[i].col == obY:
            break
        if path[i+1].row == (path[i].row + 1) % x:
            instructions.append(3)
        if path[i+1].row == (path[i].row - 1) % x:
            instructions.append(1)
        if path[i+1].col == (path[i].col + 1) % x:
            instructions.append(4)
        if path[i+1].col == (path[i].col - 1) % x:
            instructions.append(2)
    return instructions


def closestObjective(map):
    objectives = map.getObjectives()
    closest = None
    dis = 9999
    for i in range(len(objectives)-1):
        distance, s = minDist(map.grid, map.apX, map.apY, objectives[i][0], objectives[i][1], map.size)
        if closest == None:
            closest = objectives[i]
            dis = distance
        else:
            if distance < dis:
                closest = objectives[i]
                dis = distance
    if closest == None:
        return objectives[len(objectives)-1]
    return closest

def closestMoves(map):
    closest = closestObjective(map)
    distance, s = minDist(map.grid, map.apX, map.apY, closest[0], closest[1], map.size)
    return getPath(s, closest[0], closest[1], map.size)


def fullPath(map):
    fullMoves = []
    objectives = map.getObjectives()
    posX, posY = map.apX, map.apY
    for i in range(len(objectives)):
        closest = closestObjective(posX, posY, map)
        distance, s = minDist(map.grid, posX, posY, closest[0], closest[1], map.size)
        instructions = getPath(s, closest[0], closest[1], map.size)
        for inst in instructions:
            fullMoves.append(inst)
        posX, posY = closest[0], closest[1]
        if(posX == map.opX and posY == map.opY):
            fullMoves.append(5)
    return fullMoves

def printMap(m):
    for row in m:
        print(row)

def loadMaps():
    with open('maps.json', 'r') as f:
        a = json.load(f)
    return copy.deepcopy(a)

def writeMaps(maps):
    with open('maps.json', 'w') as f:
        a = json.dump(maps, f, separators=(',', ': '), indent=4)

def reshape(grid):
    newGrid = np.array(grid)
    newGrid = newGrid.reshape(400)
    return newGrid

def generateMaps(mapNum):
    maps = []
    move = []
    while len(maps) < mapNum:
        game = Game()
        game.createGrid(20)
        try:
            while len(game.map.getObjectives()) > 0:
                for i in closestMoves(game.map):
                    move.append(i)
                    maps.append(reshape(game.map.grid))
                    game.makeMove(i)
                    if (game.map.opX == game.map.apX and game.map.opY == game.map.apY):
                        break
                if (game.map.opX == game.map.apX and game.map.opY == game.map.apY):
                    break
        except Exception:
            continue
    return maps, move


def main():
    maps, moves = generateMaps(1)
    print(moves)
    newGame = Game()
    newGame.loadGrid(maps[0].reshape(20,20))

    # maps = loadMaps()
    # newGame = Game()
    # newGame.createGrid(20)
    graphicsHandler = Graphics(35, newGame.map)
    graphicsHandler.createBoard()
    #
    #
    while 1:
        graphicsHandler.board.pack()
        for i in closestMoves(newGame.map):
            newGame.makeMove(i)
            print(i)
            time.sleep(0.3)
            graphicsHandler.updateBoard()
            graphicsHandler.master.update()
        if (newGame.map.opX == newGame.map.apX and newGame.map.opY == newGame.map.apY):
            break




if __name__ == '__main__':
    main()
