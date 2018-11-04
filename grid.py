import tkinter as tk
import numpy as np

master = tk.Tk()
master.wm_title("gridWorld")

apX, apY = np.random.randint(0, 20), np.random.randint(0, 20)
grid =[]
Width = 35
x = 20
grid = [[0 for row in range(x)] for col in range(x)]
grid[apX][apY] = 1
gridBoard = [[0 for row in range(x)] for col in range(x)]
board = tk.Canvas(master, width=x*Width, height=x*Width)

for i in range(x):
    for j in range(x):
        if grid[i][j] == 0:
            gridBoard[i][j] = board.create_rectangle(i*Width, j*Width, (i+1)*Width, (j+1)*Width, fill="white", width=1)
        elif grid[i][j] == 1:
            gridBoard[i][j] = board.create_rectangle(i*Width, j*Width, (i+1)*Width, (j+1)*Width, fill="red", width=1)

def updateBoard():
    for i in range(x):
        for j in range(x):
            if grid[i][j] == 0:
                board.itemconfig(gridBoard[i][j], fill="white")
            elif grid[i][j] == 1:
                board.itemconfig(gridBoard[i][j], fill="red")

def key(event):
    global apX
    global apY
    kp = event.char
    if kp == 'a':
        grid[apX][apY] = 0
        apX = apX - 1
        grid[apX][apY] = 1
    elif kp == 'd':
        grid[apX][apY] = 0
        apX = apX + 1
        grid[apX][apY] = 1
    elif kp == 'w':
        grid[apX][apY] = 0
        apY = apY - 1
        grid[apX][apY] = 1
    elif kp == 's':
        grid[apX][apY] = 0
        apY = apY + 1
        grid[apX][apY] = 1
    updateBoard()
def callback(event):
    board.focus_set()
    print("clicked at", event.x, event.y)
board.bind("<Key>", key)
board.bind("<Button-1>", callback)
board.pack()
master.mainloop()
