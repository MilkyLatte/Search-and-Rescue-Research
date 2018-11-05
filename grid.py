import tkinter as tk
import numpy as np

master = tk.Tk()
master.wm_title("gridWorld")

apX, apY = np.random.randint(0, 20), np.random.randint(0, 20)
opX, opY = np.random.randint(0, 20), np.random.randint(0, 20)
grid =[]
Width = 35
x = 20
grid = [[0 for row in range(x)] for col in range(x)]
grid[apX][apY] = 1
grid[opX][opY] = 4
gridBoard = [[0 for row in range(x)] for col in range(x)]
board = tk.Canvas(master, width=x*Width, height=x*Width)
moveCounter = 0

gameOver = False

for i in range(x):
    for j in range(x):
        if (i != apX and j != apY and i != opX and j != opY):
            randomMap = np.random.randint(0, 2)
            if randomMap == 1:
                grid[i][j] = 2


def smoothMap():
    global grid
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
    grid = gridCopy

for i in range(15):
    smoothMap()
for i in range(x):
    for j in range(x):
        if (i != apX and j != apY and grid[i][j] != 2):
            objective = np.random.randint(0, 30)
            if objective == 1:
                grid[i][j] = 3
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

def key(event):
    global apX
    global apY
    global moveCounter
    kp = event.char
    #smoothMap()
    if kp == 'a':
        if grid[(apX-1)%20][apY] != 2:
            grid[apX][apY] = 0
            apX = (apX - 1)%x
            grid[apX][apY] = 1
            moveCounter += 1
    elif kp == 'd':
        if grid[(apX+1)%20][apY] != 2:
            grid[apX][apY] = 0
            apX = (apX + 1)%x
            grid[apX][apY] = 1
            moveCounter += 1
    elif kp == 'w':
        if grid[apX][(apY-1)%20] != 2:
            grid[apX][apY] = 0
            apY = (apY - 1)%x
            grid[apX][apY] = 1
            moveCounter += 1
    elif kp == 's':
        if grid[apX][(apY+1)%20] != 2:
            grid[apX][apY] = 0
            apY = (apY + 1)%x
            grid[apX][apY] = 1
            moveCounter += 1
    elif kp == 'c' and apX == opX and apY == opY:
        gameOver()


def gameOver():
    global gameOver
    gameOver = True
    print(moveCounter)

def callback(event):
    board.focus_set()
    print("clicked at", event.x, event.y)


while(1):
    if gameOver == True:
        break
    updateBoard()
    board.bind("<Key>", key)
    board.bind("<Button-1>", callback)
    board.pack()
    master.update_idletasks()
    master.update()
