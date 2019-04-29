import numpy as np
import tkinter as tk
import time
import pandas as pd
import copy
import json
from ortools.constraint_solver import pywrapcp

WALL = 0
COIN = 1
OBJECTIVE = 0.8
AGENT = 0.5
SPACE = 0.6
TRAIL = 0.4

def checkerPathern(gridSize):
    grid = [[0 for row in range(gridSize)] for col in range(gridSize)]
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if row%2 != 0 and col%2 == 0:
                grid[row][col] = 1
            # elif row%2 != 0 and col%2 == 0:
            #     grid[row][col] = 1
    return grid



def smoothMap(grid, x):
    gridCopy = grid
    for i in range(x):
        for j in range(x):
            if(grid[i][j]==WALL):
                diagonals = 0
                axis = 0
                if grid[(i+1)%x][j]==SPACE:
                    axis += 1
                if grid[(i-1)%x][j]==SPACE:
                    axis+=1
                if grid[i][(j+1)%x]==SPACE:
                    axis+=1
                if grid[i][(j-1)%x]==SPACE:
                    axis+=1
                if grid[(i+1)%x][(j+1)%x]==SPACE:
                    diagonals+=1
                if grid[(i+1)%x][(j-1)%x]==SPACE:
                    diagonals+=1
                if grid[(i-1)%x][(j+1)%x]==SPACE:
                    diagonals+=1
                if grid[(i-1)%x][(j-1)%x]==SPACE:
                    diagonals+=1
                if diagonals == 1 or axis == 4 or axis == 2:
                    gridCopy[i][j] = SPACE
            if(grid[i][j]==SPACE):
                diagonals = 0
                axis = 0
                if grid[(i+1)%x][j]==WALL:
                    axis += 1
                if grid[(i-1)%x][j]==WALL:
                    axis+=1
                if grid[i][(j+1)%x]==WALL:
                    axis+=1
                if grid[i][(j-1)%x]==WALL:
                    axis+=1
                if grid[(i+1)%x][(j+1)%x]==WALL:
                    diagonals+=1
                if grid[(i+1)%x][(j-1)%x]==WALL:
                    diagonals+=1
                if grid[(i-1)%x][(j+1)%x]==WALL:
                    diagonals+=1
                if grid[(i-1)%x][(j-1)%x]==WALL:
                    diagonals+=1
                if diagonals == 1 or axis == 4:
                    gridCopy[i][j] = WALL
    return grid

class Game():
    def __init__(self):
        self.moveCounter = 0
        self.objectiveCounter = 0
        self.map = []
        self.testMaps = self.loadTestMaps()

    def loadTestMaps(self):
        with open('Test.json', 'r') as f:
            a = json.load(f)
        return copy.deepcopy(a)

    def testMaze(self, gridSize, objNum):
        grid = [[SPACE for row in range(gridSize)] for col in range(gridSize)]
        v = 1
        apX = np.random.randint(v, gridSize-(v+1))
        apY = np.random.randint(v, gridSize-(v+1))

        for row in range(len(grid)):
            for col in range(len(grid)):
                if row <= v-1 or row >= gridSize-v:
                    grid[row][col] = WALL
                if col <= v-1 or col >= gridSize-v:
                    grid[row][col] = WALL       
        maxObjectives = 0
        for i in range(objNum):
            while 1:
                x = np.random.randint(v, gridSize-(v+1))
                y = np.random.randint(v, gridSize-(v+1))
                if x != apX and y != apY and grid[x][y] != COIN:
                    grid[x][y] = COIN
                    maxObjectives += 1
                    break

        grid[apX][apY] = AGENT



        self.map = Grid(apX, apY, grid, maxObjectives, gridSize, [(apX, apY)])
        _, self.minMoves = brute_force(self.map)


    def mazeLevelOne(self, gridSize, objNum):
        grid = [[0 for row in range(gridSize)] for col in range(gridSize)]
        visited = Stack()
        pos = (np.random.randint(0, gridSize), np.random.randint(0, gridSize))
        while 1:
            attempted = []
            while 1:
                direction = np.random.randint(0, 4)
                # DOWN
                if direction == 0 and not(direction in attempted):
                    if pos[0]+2 > gridSize-1 or grid[pos[0]+2][pos[1]] == SPACE or pos[0]+1 > gridSize-1:
                        attempted.append(direction)
                        continue
                    else:
                        visited.push(pos)
                        grid[pos[0]+1][pos[1]] = SPACE
                        grid[pos[0]+2][pos[1]] = SPACE
                        pos = (pos[0]+2 , pos[1])
                        break
                # UP
                if direction == 1 and not(direction in attempted):
                    if pos[0]-2 < 0 or grid[pos[0]-2][pos[1]] == SPACE or pos[0]-1 < 0:
                        attempted.append(direction)
                        continue
                    else:
                        visited.push(pos)
                        grid[pos[0]-1][pos[1]] = SPACE
                        grid[pos[0]-2][pos[1]] = SPACE
                        pos = (pos[0]-2 , pos[1])
                        break
                # RIGHT
                if direction == 2 and not(direction in attempted):
                    if pos[1]+2 > gridSize-1 or grid[pos[0]][pos[1]+2] == SPACE or pos[1]+1 > gridSize-1:
                        attempted.append(direction)
                        continue
                    else:
                        visited.push(pos)
                        grid[pos[0]][pos[1]+1] = SPACE
                        grid[pos[0]][pos[1]+2] = SPACE
                        pos = (pos[0] , pos[1]+2)
                        break
                # LEFT
                if direction == 3 and not(direction in attempted):
                    if pos[1]-2 < 0 or grid[pos[0]][pos[1]-2] == SPACE or pos[1]+1 < 0:
                        attempted.append(direction)
                        continue
                    else:
                        visited.push(pos)
                        grid[pos[0]][pos[1]-1] = SPACE
                        grid[pos[0]][pos[1]-2] = SPACE
                        pos = (pos[0] , pos[1]-2)
                        break
                if len(attempted) == 4:
                    pos = visited.pop()
                    break
            if len(visited.stack) == 0:
                break
        while 1:
            apX, apY = np.random.randint(0, gridSize), np.random.randint(0, gridSize)
            if grid[apX][apY] != WALL:
                break
        grid[apX][apY] = AGENT

        maxObjectives = 0
        for i in range(objNum):
            while 1:
                x = np.random.randint(0, gridSize-1)
                y = np.random.randint(0, gridSize-1)
                if x != apX and y != apY and grid[x][y] != WALL:
                    grid[x][y] = COIN
                    maxObjectives += 1
                    break

        # for i in range(gridSize):
        #     for j in range(gridSize):
        #         if (i != apX and j != apY and grid[i][j] != WALL):
        #             objective = np.random.randint(0, 30)
        #             if objective == 1:
        #                 if grid[i][j] == WALL:
        #                     continue
        #                 grid[i][j] = COIN
        self.map = Grid(apX, apY, grid, maxObjectives, gridSize, [(apX, apY)])
        _, self.minMoves = brute_force(self.map)


    def createGrid(self, gridSize, objNum):
        grid = [[SPACE for row in range(gridSize)] for col in range(gridSize)]
        apX = np.random.randint(1, gridSize-2)
        apY = np.random.randint(1, gridSize-2)

        grid[apX][apY] = AGENT
        maxObjectives = 0
        for i in range(gridSize):
            for j in range(gridSize):
                if (i != apX and j != apY):
                    # CHANGE THIS TO GENERATE MAPS WITH MORE WALLS
                    randomMap = np.random.randint(0, 10)
                    if randomMap == 1:
                        grid[i][j] = WALL
        for i in range(0):
            grid = smoothMap(grid, gridSize)
        for i in range(objNum):
            while 1:
                x = np.random.randint(1, gridSize-1)
                y = np.random.randint(1, gridSize-1)
                if x != apX and y != apY and grid[x][y] != WALL:
                    grid[x][y] = COIN
                    maxObjectives += 1
                    break
        for row in range(len(grid)):
            for col in range(len(grid)):
                if row == 0 or row == gridSize-1:
                    grid[row][col] = WALL
                if col == 0 or col == gridSize-1:
                    grid[row][col] = WALL
        self.map = Grid(apX, apY, grid, maxObjectives, gridSize, [(apX, apY)])
        _, self.minMoves = brute_force(self.map)


    def loadGrid(self, grid):
        apX, apY, maxObjectives = 0, 0, 0
        for row in range(len(grid)):
            for col in range(len(grid)):
                if grid[row][col] == AGENT:
                    apX = row
                    apY = col
                elif grid[row][col] == COIN:
                    maxObjectives += 1
        self.map = Grid(apX, apY, grid, maxObjectives, len(grid), [(apX, apY)])
        _, self.minMoves = brute_force(self.map)


    def appendToTrail(self):
        self.map.grid[self.map.trail[len(self.map.trail)-1][0]][self.map.trail[len(self.map.trail)-1][1]] = SPACE
        # Change this value to use longer trails
        if len(self.map.trail) == 4:
            self.map.trail.pop(len(self.map.trail)-1)
            self.map.trail.insert(0, (self.map.apX, self.map.apY))
        else:
            self.map.trail.insert(0, (self.map.apX, self.map.apY))
        self.updateTrail()


    def updateTrail(self):
        self.map.grid[self.map.trail[0][0]][self.map.trail[0][1]] = AGENT
        for i in range(1, len(self.map.trail)):
            if self.map.grid[self.map.trail[i][0]][self.map.trail[i][1]] == AGENT:
                continue
            self.map.grid[self.map.trail[i][0]][self.map.trail[i][1]] = TRAIL

    def makeMove(self, move):
        self.moveCounter += 1
        previous = False
        # Left
        if move == 0:
            if self.map.grid[(self.map.apX-1)%self.map.size][self.map.apY] != WALL:
                self.map.grid[self.map.apX][self.map.apY] = SPACE
                if self.map.grid[(self.map.apX-1)%self.map.size][self.map.apY] == COIN:
                    self.objectiveCounter += 1
                if self.map.grid[(self.map.apX-1)%self.map.size][self.map.apY] == TRAIL:
                    previous = True
                self.map.apX = (self.map.apX - 1)%self.map.size
                self.map.grid[self.map.apX][self.map.apY] = AGENT
                self.appendToTrail()
        # Up
        elif move == 1:
            if self.map.grid[self.map.apX][(self.map.apY-1)%self.map.size] != WALL:
                self.map.grid[self.map.apX][self.map.apY] = SPACE
                if self.map.grid[self.map.apX][(self.map.apY-1)%self.map.size] == COIN:
                    self.objectiveCounter += 1
                if self.map.grid[self.map.apX][(self.map.apY-1)%self.map.size] == TRAIL:
                    previous = True
                self.map.apY = (self.map.apY - 1)%self.map.size
                self.map.grid[self.map.apX][self.map.apY] = AGENT
                self.appendToTrail()
        # Right
        elif move == 2:
            if self.map.grid[(self.map.apX+1)%self.map.size][self.map.apY] != WALL:
                self.map.grid[self.map.apX][self.map.apY] = SPACE
                if self.map.grid[(self.map.apX+1)%self.map.size][self.map.apY] == COIN:
                    self.objectiveCounter += 1
                if self.map.grid[(self.map.apX+1)%self.map.size][self.map.apY] == TRAIL:
                    previous = True
                self.map.apX = (self.map.apX + 1)%self.map.size
                self.map.grid[self.map.apX][self.map.apY] = AGENT
                self.appendToTrail()
        # Down
        elif move == 3:
            if self.map.grid[self.map.apX][(self.map.apY+1)%self.map.size] != WALL:
                self.map.grid[self.map.apX][self.map.apY] = SPACE
                if self.map.grid[self.map.apX][(self.map.apY+1)%self.map.size] == COIN:
                    self.objectiveCounter += 1
                if self.map.grid[self.map.apX][(self.map.apY+1)%self.map.size] == TRAIL:
                    previous = True
                self.map.apY = (self.map.apY + 1)%self.map.size
                self.map.grid[self.map.apX][self.map.apY] = AGENT
                self.appendToTrail()
        return previous


    def getGameState(self):
        return self.map.grid, self.moveCounter, self.objectiveCounter






class Grid():
    def __init__(self, apX, apY, grid, maxObjectives, size, trail):
        self.size = size
        self.grid = grid
        self.apX = apX
        self.apY = apY
        self.maxObjectives = maxObjectives
        self.fullPath = []
        self.trail = trail

    def getObjectives(self):
        objectives = []
        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row][col] == COIN:
                    objectives.append((row, col))
        return objectives

    def getCentricPosition(self):
        centerX = self.size - self.apX
        centerY = self.size - self.apY

        padded = self.size * 2

        grid = [[WALL for row in range(padded)] for col in range(padded)]

        for row in range(self.size):
            for col in range(self.size):
                grid[row + centerX][col + centerY] = self.grid[row][col]


        return Grid(self.size, self.size, grid, 0, len(grid), [(self.size, self.size)])






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
                if self.map.grid[i][j] == SPACE:
                    self.gridBoard[i][j] = self.board.create_rectangle(i*self.width, j*self.width, (i+1)*self.width, (j+1)*self.width, fill="white", width=1)
                elif self.map.grid[i][j] == AGENT:
                    self.gridBoard[i][j] = self.board.create_rectangle(i*self.width, j*self.width, (i+1)*self.width, (j+1)*self.width, fill="green", width=1)
                elif self.map.grid[i][j] == WALL:
                    self.gridBoard[i][j] = self.board.create_rectangle(i*self.width, j*self.width, (i+1)*self.width, (j+1)*self.width, fill="black", width=1)
                elif self.map.grid[i][j] == COIN:
                    self.gridBoard[i][j] = self.board.create_rectangle(i*self.width, j*self.width, (i+1)*self.width, (j+1)*self.width, fill="yellow", width=1)

    def updateBoard(self):
        for i in range(self.map.size):
            for j in range(self.map.size):
                if self.map.grid[i][j] == SPACE:
                    self.board.itemconfig(self.gridBoard[i][j], fill="white")
                elif self.map.grid[i][j] == AGENT:
                    self.board.itemconfig(self.gridBoard[i][j], fill="green")
                elif self.map.grid[i][j] == TRAIL:
                    self.board.itemconfig(self.gridBoard[i][j], fill="blue")
                elif self.map.grid[i][j] == WALL:
                    self.board.itemconfig(self.gridBoard[i][j], fill="black")
                elif self.map.grid[i][j] == COIN:
                    self.board.itemconfig(self.gridBoard[i][j], fill="yellow")


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

from itertools import permutations

def brute_force(map):
    objectives = map.getObjectives()
    if len(objectives) == 0:
        return None, 0
    objectives.insert(0, (map.apX, map.apY))
    distance_matrix = [[0 for col in range(len(objectives))] for row in range(len(objectives))]
    
    index_name = [(i) for i in range(1, len(objectives))]
    for i in range(len(objectives)):
        for j in range(i, (len(objectives))):
            if i == j:
                continue
            else:
                dist, s = minDist(map.grid, objectives[i][0], objectives[i][1], objectives[j][0], objectives[j][1], map.size)
                distance_matrix[j][i] = dist
                distance_matrix[i][j] = dist
    
    perms= list(permutations(index_name, len(index_name)))
    winning_perm = []
    distance = 1000
    #pre = time.time()
    for perm in perms:
        d_counter = distance_matrix[0][perm[0]]
        current = perm[0]
        for node in perm:
            if node == current:
                continue
            d_counter += distance_matrix[current][node]
            if d_counter > distance:
                break
            current = node
        if d_counter < distance:
            distance = d_counter
            winning_perm = perm
    
   # pos = time.time()
   # print(pos-pre)  

    return objectives[winning_perm[0]], distance


            


    
def tsp_solver(map):
    objectives = map.getObjectives()
    objectives.insert(0, (map.apX, map.apY))
    distance_matrix = [[0 for col in range(len(objectives))] for row in range(len(objectives))]
    
    index_name = [str(i) for i in range(len(objectives))]
    for i in range(len(objectives)):
        for j in range(i, (len(objectives))):
            if i == j:
                continue
            else:
                dist, s = minDist(map.grid, objectives[i][0], objectives[i][1], objectives[j][0], objectives[j][1], map.size)
                distance_matrix[j][i] = dist
                distance_matrix[i][j] = dist
    # index_name.append("dummy")

    tsp_size = len(index_name)
    num_routes = 1
    depot = 0
    move = None
    if tsp_size > 0:
        routing = pywrapcp.RoutingModel(tsp_size, num_routes, depot)
        search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
        dist_callback = create_distance_callback(distance_matrix)
        routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)
    
        assignment = routing.SolveWithParameters(search_parameters)
        
        if assignment:
            route_number = 0
            index = routing.Start(route_number)
            index = assignment.Value(routing.NextVar(index))
            next_move = routing.IndexToNode(index)

            if index_name[next_move] == 'dummy':
                index = assignment.Value(routing.NextVar(index))
                next_move = routing.IndexToNode(index)
                
            move = objectives[next_move]

    return move

def create_distance_callback(dist_matrix):
    def distance_callback(from_node, to_node):
        return int(dist_matrix[from_node][to_node])
    return distance_callback

    # distance_matrix = Game.

def minDist(grid, sourceX, sourceY, targetX, targetY, x):
    source = Node(sourceX, sourceY, 0, -1, -1)
    visited = [[False for row in range(x)] for col in range(x)]
    for row in range(x):
        for col in range(x):
            if grid[row][col] == WALL:
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

    return -1, []


def getPath(stack, obX, obY, x, grid):
    if len(stack.stack) == 0:
        return []
    path = []
    currentPosition = stack.pop()

    while(1):
        if currentPosition.row == obX and currentPosition.col == obY:
            break
        if len(stack.stack) == 0:
            return []
        try:
            currentPosition = stack.pop()
        except Exception:
            print("HERE")
            return -1


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
            instructions.append(2)
        elif path[i+1].row == (path[i].row - 1) % x:
            instructions.append(0)
        elif path[i+1].col == (path[i].col + 1) % x:
            instructions.append(3)
        elif path[i+1].col == (path[i].col - 1) % x:
            instructions.append(1)
    return instructions


def closestObjective(map):
    objectives = map.getObjectives()
    closest = None
    dis = 9999
    for i in range(len(objectives)):
        distance, s = minDist(map.grid, map.apX, map.apY, objectives[i][0], objectives[i][1], map.size)
        if closest == None:
            closest = objectives[i]
            dis = distance
        else:
            if distance < dis:
                closest = objectives[i]
                dis = distance
    # if closest == None:
    #     return objectives[len(objectives)-1]
    return closest

def closestMoves(map):
    closest, _ = brute_force(map)
    if closest == None:
        return []
    distance, s = minDist(map.grid, map.apX, map.apY, closest[0], closest[1], map.size)
    return getPath(s, closest[0], closest[1], map.size, map)



def printMap(m):
    for row in m:
        print(row)

def loadMaps():
    with open('maps.json', 'r') as f:
        a = json.load(f)
    return copy.deepcopy(a)

def writeMaps(maps):
    with open('Train.json', 'w') as f:
        a = json.dump(maps, f, separators=(',', ': '), indent=4)

def reshape(grid):
    newGrid = np.array(grid)
    newGrid = newGrid.reshape(576)
    return np.ndarray.tolist(newGrid)

def isValidMap(map):
    for i in map.getObjectives():
        d, s = minDist(map.grid, map.apX, map.apY, i[0], i[1], map.size)
        try:
            if len(s.stack) == 0:
                return False
        except Exception:
            return False
    return True




def generateMaps(mapNum, mapType):
    maps = {
        "maps": [],
        "moves": []
        }
    gameCounter = 0
    while gameCounter < mapNum:
        game = Game()
        if mapType == 0:
            game.testMaze(12, 1)
            equalMap = False
            for i in game.testMaps["short"]:
                if np.array_equal(np.array(i).reshape(12,12), game.map.grid):
                    equalMap = True
                    print("REPEATED")
                    break
            if equalMap:
                continue
        if mapType == 1:
            game.testMaze(12, 7)
            equalMap = False
            for i in game.testMaps["tsp"]:
                if np.array_equal(np.array(i).reshape(12,12), game.map.grid):
                    equalMap = True
                    break
            if equalMap:
                continue
        if mapType == 2:
            game.createGrid(12, 7)
            equalMap = False
            for i in game.testMaps["saltpepper"]:
                if np.array_equal(np.array(i).reshape(12,12), game.map.grid):
                    equalMap = True
                    break
            if equalMap:
                continue
        if mapType == 3:
            game.mazeLevelOne(12, 7)
            equalMap = False
            for i in game.testMaps["maze"]:
                if np.array_equal(np.array(i).reshape(12,12), game.map.grid):
                    equalMap = True
                    break
            if equalMap:
                continue
        if not isValidMap(game.map):
            continue
        while len(game.map.getObjectives()) > 0:
            closest = closestMoves(game.map)
            if closest == -1:
                break
            for i in closest:
                maps["moves"].append(i)
                maps["maps"].append(list(reshape(game.map.getCentricPosition().grid)))
                game.makeMove(i)
                if (len(game.map.getObjectives()) == 0):
                    break
            if (len(game.map.getObjectives()) == 0):
                gameCounter += 1
                break

    writeMaps(maps)
    return maps

def generateTestMaps(mapNum):
    maps = {
        "short": [],
        "tsp": [],
        "saltpepper":[],
        "maze": []
        }
    gameCounter = 0
    for i in range(4):
        while gameCounter < mapNum:
            game = Game()
            if i == 0:
                game.testMaze(12, 1)
                if not isValidMap(game.map):
                    continue
                maps["short"].append(list(reshape(game.map.grid)))
            if i == 1:
                game.testMaze(12, 7)
                if not isValidMap(game.map):
                    continue
                maps["tsp"].append(list(reshape(game.map.grid)))
            if i == 2:
                game.mazeLevelOne(12, 7)
                if not isValidMap(game.map):
                    continue
                maps["saltpepper"].append(list(reshape(game.map.grid)))
            if i == 3:
                game.createGrid(12, 7)
                if not isValidMap(game.map):
                    continue
                maps["maze"].append(list(reshape(game.map.grid)))
            gameCounter += 1
        gameCounter = 0
    writeMaps(maps)
    return maps
    

def main():
    # generateTestMaps(100)
    generateMaps(1000, 2)




if __name__ == '__main__':
    main()
