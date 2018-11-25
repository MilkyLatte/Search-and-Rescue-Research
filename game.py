import numpy as np
import tkinter as tk
import json
import time
import copy

master = tk.Tk()
master.wm_title("gridWorld")
x = 20
Width = 35
objectiveCounter = 0
moveCounter = 0


def createWorld(x):
    apX, apY = np.random.randint(0, x), np.random.randint(0, x)
    opX, opY = np.random.randint(0, x), np.random.randint(0, x)
    grid =[]
    grid = [[0 for row in range(x)] for col in range(x)]
    grid[apX][apY] = 1
    grid[opX][opY] = 4
    maxObjectives = 0
    for i in range(x):
        for j in range(x):
            if (i != apX and j != apY and i != opX and j != opY):
                randomMap = np.random.randint(0, 2)
                if randomMap == 1:
                    grid[i][j] = 2
    for i in range(15):
        grid = smoothMap(grid)
    for i in range(x):
        for j in range(x):
            if (i != apX and j != apY and grid[i][j] != 2):
                objective = np.random.randint(0, 30)
                if objective == 1:
                    grid[i][j] = 3
                    maxObjectives += 1
    return apX, apY, opX, opY, grid, maxObjectives


def createBoard():
    gridBoard = [[0 for row in range(x)] for col in range(x)]
    board = tk.Canvas(master, width=x*Width, height=x*Width)
    for i in range(x):
        for j in range(x):
            if grid[i][j] == 0:
                gridBoard[i][j] = board.create_rectangle(i*Width, j*Width, (i+1)*Width, (j+1)*Width, fill="white", width=1)
            elif grid[i][j] == 1:
                gridBoard[i][j] = board.create_rectangle(i*Width, j*Width, (i+1)*Width, (j+1)*Width, fill="green", width=1)
            elif grid[i][j] == 2:
                gridBoard[i][j] = board.create_rectangle(i*Width, j*Width, (i+1)*Width, (j+1)*Width, fill="black", width=1)
            elif grid[i][j] == 3:
                gridBoard[i][j] = board.create_rectangle(i*Width, j*Width, (i+1)*Width, (j+1)*Width, fill="yellow", width=1)
            elif grid[i][j] == 4:
                gridBoard[i][j] = board.create_rectangle(i*Width, j*Width, (i+1)*Width, (j+1)*Width, fill="purple", width=1)
    for i in range(x):
        for j in range(x):
            if grid[i][j] == 0:
                gridBoard[i][j] = board.create_rectangle(i*Width, j*Width, (i+1)*Width, (j+1)*Width, fill="white", width=1)
            elif grid[i][j] == 1:
                gridBoard[i][j] = board.create_rectangle(i*Width, j*Width, (i+1)*Width, (j+1)*Width, fill="green", width=1)
            elif grid[i][j] == 2:
                gridBoard[i][j] = board.create_rectangle(i*Width, j*Width, (i+1)*Width, (j+1)*Width, fill="black", width=1)
            elif grid[i][j] == 3:
                gridBoard[i][j] = board.create_rectangle(i*Width, j*Width, (i+1)*Width, (j+1)*Width, fill="yellow", width=1)
            elif grid[i][j] == 4:
                gridBoard[i][j] = board.create_rectangle(i*Width, j*Width, (i+1)*Width, (j+1)*Width, fill="purple", width=1)
    return board, gridBoard


def smoothMap(grid):
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

def key(event):
    global apX
    global apY
    global moveCounter
    global objectiveCounter
    kp = event.char
    if kp == 'a':
        if grid[(apX-1)%20][apY] != 2:
            grid[apX][apY] = 0
            if grid[(apX-1)%x][apY] == 3:
                objectiveCounter += 1
            apX = (apX - 1)%x
            grid[apX][apY] = 1
            moveCounter += 1
    elif kp == 'w':
        if grid[apX][(apY-1)%20] != 2:
            grid[apX][apY] = 0
            if grid[apX][(apY-1)%x] == 3:
                objectiveCounter += 1
                print()
            apY = (apY - 1)%x
            grid[apX][apY] = 1
            moveCounter += 1
    elif kp == 'd':
        if grid[(apX+1)%20][apY] != 2:
            grid[apX][apY] = 0
            if grid[(apX+1)%x][apY] == 3:
                objectiveCounter += 1
            apX = (apX + 1)%x
            grid[apX][apY] = 1
            moveCounter += 1
    elif kp == 's':
        if grid[apX][(apY+1)%20] != 2:
            grid[apX][apY] = 0
            if grid[apX][(apY+1)%x] == 3:
                objectiveCounter += 1
            apY = (apY + 1)%x
            grid[apX][apY] = 1
            moveCounter += 1
    elif kp == 'c' and apX == opX and apY == opY:
        gameOver()
    elif kp == 'x':
        print(apX, apY)


def callback(event):
    board.focus_set()


def updateBoard():
    if apX != opX and apY != opY:
        grid[opX][opY] = 4
    for i in range(x):
        for j in range(x):
            if grid[i][j] == 0:
                board.itemconfig(gridBoard[i][j], fill="white")
            elif grid[i][j] == 1:
                board.itemconfig(gridBoard[i][j], fill="green")
            elif grid[i][j] == 4:
                board.itemconfig(gridBoard[i][j], fill="purple")

def loadMaps():
    with open('maps.json', 'r') as f:
        a = json.load(f)
    return copy.deepcopy(a)

def writeMaps(maps):
    with open('maps.json', 'w') as f:
        a = json.dump(maps, f, separators=(',', ': '), indent=4)

def gameOver():
    global gameOver
    gameOver = True
    print(moveCounter)
    print("Objectives: " + str(objectiveCounter)+ "/" + str(maxObjectives))

def userGame():
    while 1:
        if gameOver == True:
            break
        updateBoard()
        board.bind("<Key>", key)
        board.bind("<Button-1>", callback)
        board.pack()
        master.update()

def moveHandler(move):
    global apX
    global apY
    global moveCounter
    global objectiveCounter
    if move == 0:
        pass
    elif move == 1:
        if grid[(apX-1)%20][apY] != 2:
            grid[apX][apY] = 0
            if grid[(apX-1)%x][apY] == 3:
                objectiveCounter += 1
            apX = (apX - 1)%x
            grid[apX][apY] = 1
            moveCounter += 1
    elif move == 2:
        if grid[apX][(apY-1)%20] != 2:
            grid[apX][apY] = 0
            if grid[apX][(apY-1)%x] == 3:
                objectiveCounter += 1
                print()
            apY = (apY - 1)%x
            grid[apX][apY] = 1
            moveCounter += 1
    elif move == 3:
        if grid[(apX+1)%20][apY] != 2:
            grid[apX][apY] = 0
            if grid[(apX+1)%x][apY] == 3:
                objectiveCounter += 1
            apX = (apX + 1)%x
            grid[apX][apY] = 1
            moveCounter += 1
    elif move == 4:
        if grid[apX][(apY+1)%20] != 2:
            grid[apX][apY] = 0
            if grid[apX][(apY+1)%x] == 3:
                objectiveCounter += 1
            apY = (apY + 1)%x
            grid[apX][apY] = 1
            moveCounter += 1
    elif move == 5 and apX == opX and apY == opY:
        gameOver()


def randomMove():
    return np.random.randint(0, high=6)

def fullPath():
    fullMoves = []
    objectives = getObjectives()
    posX, posY = apX, apY
    for i in range(len(objectives)):
        distance, s = minDist(grid, posX, posY, objectives[i][0], objectives[i][1])
        instructions = getPath(s, objectives[i][0], objectives[i][1])
        for inst in instructions:
            fullMoves.append(inst)
        posX, posY = objectives[i][0], objectives[i][1]
        if(posX == opX and posY == opY):
            fullMoves.append(5)
    return fullMoves


def machineGameTSM():
    instructions = fullPath()
    while 1:
        if gameOver == True:
            break
        updateBoard()
        if len(instructions) > 0:
            moveHandler(instructions.pop(0))
        time.sleep(0.1)
        board.pack()
        master.update()

def processLoadedMap():
    apX, apY, opX, opY, maxObjectives = 0, 0, 0, 0, 0
    for row in range(x):
        for col in range(x):
            if grid[row][col] == 1:
                apX = row
                apY = col
            elif grid[row][col] == 4:
                opX = row
                opY = col
            elif grid[col][row] == 3:
                maxObjectives += 1
    return apX, apY, opX, opY, maxObjectives

def getObjectives():
    objectives = []
    for row in range(x):
        for col in range(x):
            if grid[row][col] == 3:
                objectives.append((row, col))
    objectives.append((opX, opY))
    return objectives


def printMap(m):
    for row in m:
        print(row)

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

def minDist(grid, sourceX, sourceY, targetX, targetY):
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

def getPath(stack, obX, obY):
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

def generateMaps(mapNum):
    global apX, apY, opX, opY, grid, maxObjectives, board, gridBoard
    m = []
    solution = []
    while len(m) < mapNum:
        apX, apY, opX, opY, grid, maxObjectives = createWorld(x)
        try:
            s = fullPath()
            m.append(grid)
            solution.append(s)
        except Exception:
            continue
    return m, solution


def main():
    global apX, apY, opX, opY, grid, maxObjectives, board, gridBoard, maps
    maps = loadMaps()
    print(maps)
    while 1:
        mapOption = input("To play a new map press n followed by enter, to load a map press l followed by enter, g to generate new maps\n")
        while 1:
            if mapOption == 'n':
                apX, apY, opX, opY, grid, maxObjectives = createWorld(x)
                board, gridBoard = createBoard()
                initialGrid = copy.deepcopy(grid)
                break
            elif mapOption == 'l':
                whichMap = input("which map index would you like to load?\n")
                try:
                    whichMap = int(whichMap)
                    try:
                        grid = maps["validMaps"][whichMap]
                        apX, apY, opX, opY, maxObjectives = processLoadedMap()
                        board, gridBoard = createBoard()
                        initialGrid = copy.deepcopy(grid)
                        break
                    except Exception:
                        print("try a different index")
                except Exception:
                    print("invalid input try again")
            elif mapOption == 'g':
                while 1:
                    mapNum = input("how many maps?")
                    try:
                        mapNum = int(mapNum)
                        o, s = generateMaps(mapNum)
                        print("here")
                        for j in o:
                            maps["validMaps"].append(j)
                        for j in s:
                            maps["solutions"].append(j)
                        writeMaps(maps)
                        print("DONE")
                        return 0
                    except Exception:
                        print("try another number")
        break

    while 1:
        board.pack()
        machineGameTSM()
        save = input("Do you want to save this map?\n")
        while(1):
            if save == 'y':
                maps["validMaps"].append(initialGrid)
                writeMaps(maps)
                break
            elif save == 'n':
                break

        break


if __name__ == "__main__":
    main()
