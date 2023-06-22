# *********************************************
# * Three layer Feed Forward Neural Network   *
# * using Rumelhart's Back Propagation        *
# *                  by                       *
# *	         Himanshu Mazumdar                *
# *	 Tested with XOR DATA input output        *
# *	 Date:- 7-August-2022                     *
# *********************************************
import tkinter as tk
from tkinter import Canvas
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import asksaveasfile
import math
import random
from random import randint
import threading as th
import time
import NnLib as nnet

# .........................................
app = tk.Tk()
app.title("Canvas")
# wdt= app.winfo_screenwidth()#getting screen width and height of monitor
# hgt= app.winfo_screenheight()
wdt = 500
hgt = 420
# dstMx = math.sqrt(wdt*wdt+hgt*hgt)
app.geometry("%dx%d" % (wdt, hgt))  # setting tkinter window size
# app.state('zoomed')
canvas = tk.Canvas(bg="blue", width=wdt - 10, height=hgt - 120)
canvas.pack(anchor=tk.NW)
place1 = tk.Canvas(bg="orange", width=wdt - 10, height=60)
place1.pack(anchor=tk.NW, padx=5, pady=5)
place2 = tk.Canvas(bg="white", width=wdt - 10, height=60)
place2.pack(anchor=tk.NW)
place3 = tk.Canvas(bg="green", width=wdt - 10, height=60)
place3.pack(anchor=tk.NE)
count = 0
exit = 0
mag = 5.0
net = nnet.NNijk(10, 7, 5)  # ii, jj, kk
error = []


# .............................................
def set_cmd():
    global net
    global count
    count = 0
    net.count = 0
    net.ii = int(nnIn.get("1.0", END))
    net.jj = int(nnHdn.get("1.0", END))
    net.kk = int(nnOut.get("1.0", END))
    net.eta = float(nneta.get("1.0", END))
    net.init_weights(net.ii, net.jj, net.kk)
    net.load_random_weight()


# .........................................
def nnXor():
    global net
    global btnTrain
    global exit
    nm = btnTrain["text"]
    if nm == "Train":
        exit = 0
        btnTrain["text"] = "Stop"
        # net.load_random_weight()
        global count
        count = 0
        error = []
        tmr = th.Timer(0.1, trainNet)
        tmr.start()
    else:
        btnTrain["text"] = "Train"
        exit = 1


# .........................................
def trainNet():
    # while True:
    global canvas
    global count
    global net
    global exit
    global mag
    if exit == 0:
        # if count >= 0:
        net.ioflag = 1
        err = net.train_net().split(",")
        sno.delete(1.0, END)
        sno.insert(1.0, err[0])
        intErr.delete(1.0, END)
        intErr.insert(1.0, err[1])
        realErr.delete(1.0, END)
        realErr.insert(1.0, err[2])
        xn = mag * float(err[0])
        error.append(xn)
        error.append((hgt - 100) * (1 - 1.7 * float(err[2])))
        if count > 0:
            canvas.create_line(error, fill="yellow", width=3)
        # print(err)
        count += 1
        if xn > wdt:
            for ni in range(int(len(error) / 2)):
                error[ni * 2] *= 0.8
            mag *= 0.8
            canvas.create_rectangle(
                0, 0, canvas.winfo_width(), canvas.winfo_height(), fill="blue"
            )
        if count < 10000:
            tmr = th.Timer(0.01, trainNet)
            tmr.start()
        else:
            print("Over")
        # net.save_weights("NnHsm01.net")
    else:
        # exit = 0
        count = -1
        # quit()


# .............................................
def reset_cmd():
    global net
    global count
    global error
    error = []
    count = 0
    net.count = 0
    net.load_random_weight()
    canvas.create_rectangle(
        0, 0, canvas.winfo_width(), canvas.winfo_height(), fill="blue"
    )


# .............................................
def quitcmd():
    global exit
    exit += 1
    # time.sleep(5)
    if exit >= 2:
        quit()


# .............................................
def nnLoadNet():
    global net
    global error
    global count
    flnm = filedialog.askopenfilename(
        initialdir="",
        filetypes=(("net files", "*.net"), ("all files", "*.*")),
        title="Select Net File (*.net)",
    )
    net.load_weights(flnm)
    nnIn.delete(1.0, END)
    nnIn.insert(1.0, str(net.ii))
    nnHdn.delete(1.0, END)
    nnHdn.insert(1.0, str(net.jj))
    nnOut.delete(1.0, END)
    nnOut.insert(1.0, str(net.kk))
    error = []
    count = 0
    net.count = 0
    canvas.create_rectangle(
        0, 0, canvas.winfo_width(), canvas.winfo_height(), fill="blue"
    )


# .............................................
def nnSaveNet():
    global net
    flnm = asksaveasfile(
        initialfile="Untitled.txt",
        defaultextension=".net",
        filetypes=[("All Files", "*.*"), ("Text Documents", "*.txt")],
        title="Enter Net Save File Name (*.net)",
    )
    net.save_weights(flnm.name)


# .............................................
def nnLoadData():
    global net
    # flnm = filedialog.askopenfilename(("data files", "*.dat"),("all files", "*.*"))
    flnm = filedialog.askopenfilename(
        initialdir="",
        filetypes=(("data files", "*.dat"), ("all files", "*.*")),
        title="Select Data File (*.dat)",
    )
    net.load_data(flnm)
    # net.load_weights(flnm.name)


# .............................................
def nnSaveXorData():
    global net
    flnm = asksaveasfile(
        initialfile="Untitled.txt",
        defaultextension=".net",
        filetypes=[("All Files", "*.*"), ("Text Documents", "*.txt")],
        title="Enter Data Save File Name (*.dat)",
    )
    createXorData(flnm.name)


# .............................................
def createXorData(flnm):
    global net
    smx = int(smax.get("1.0", END))
    ii = int(nnIn.get("1.0", END))
    kk = int(nnOut.get("1.0", END))
    page = ""
    for s in range(smx):
        xin = []
        for i in range(ii):
            xin.append(random.randint(0, 1))
            page += str(xin[i]) + ","
        page = page[:-1] + ":"
        for k in range(kk):
            page += str(xin[k * 2] ^ xin[k * 2 + 1]) + ","
        page = page[:-1] + "\n"
    page = page[:-1]
    file = open(flnm, "w")
    file.writelines(page)
    file.close()


# .............................................
lblin = tk.Label(place1, text="NN-In", fg="black")
lblin.pack(side=tk.LEFT)
nnIn = tk.Text(place1, height=1, width=4)
nnIn.insert(tk.END, "10")
nnIn.pack(side=tk.LEFT)
lblhdn = tk.Label(place1, text="NN-Hdn", fg="black")
lblhdn.pack(side=tk.LEFT)
nnHdn = tk.Text(place1, height=1, width=4)
nnHdn.insert(tk.END, "7")
nnHdn.pack(side=tk.LEFT)
lblout = tk.Label(place1, text="NN-Out", fg="black")
lblout.pack(side=tk.LEFT)
nnOut = tk.Text(place1, height=1, width=4)
nnOut.insert(tk.END, "5")
nnOut.pack(side=tk.LEFT)
lbleta = tk.Label(place1, text="eta", fg="black")
lbleta.pack(side=tk.LEFT)
nneta = tk.Text(place1, height=1, width=4)
nneta.insert(tk.END, "0.1")
nneta.pack(side=tk.LEFT)
btnSet = tk.Button(place1, text="Set", bg="lightgray", command=set_cmd)
btnSet.pack(side=tk.LEFT)
# .......................
lbl0 = tk.Label(place2, text="SNo", fg="black")
lbl0.pack(side=tk.LEFT)
sno = tk.Text(place2, height=1, width=4)
sno.insert(tk.END, "     ")
sno.pack(side=tk.LEFT)
lbl1 = tk.Label(place2, text="IntErr", fg="black")
lbl1.pack(side=tk.LEFT)
intErr = tk.Text(place2, height=1, width=7)
intErr.insert(tk.END, "     ")
intErr.pack(side=tk.LEFT)
lbl2 = tk.Label(place2, text="Error", fg="black")
lbl2.pack(side=tk.LEFT)
realErr = tk.Text(place2, height=1, width=10)
realErr.insert(tk.END, "     ")
realErr.pack(side=tk.LEFT)
btnTrain = tk.Button(place2, text="Train", bg="lightgray", command=nnXor)
btnTrain.pack(side=tk.LEFT)
btnReset = tk.Button(place2, text="Reset", bg="lightgray", command=reset_cmd)
btnReset.pack(side=tk.LEFT)
# .............................................
lbl3 = tk.Label(place3, text="SmplMx", fg="black")
lbl3.pack(side=tk.LEFT)
smax = tk.Text(place3, height=1, width=4)
smax.insert(tk.END, "1000")
smax.pack(side=tk.LEFT)
btnLdNet = tk.Button(place3, text="Load Net", bg="lightgray", command=nnLoadNet)
btnLdNet.pack(side=tk.LEFT)
btnSvNet = tk.Button(place3, text="Save Net", bg="lightgray", command=nnSaveNet)
btnSvNet.pack(side=tk.LEFT)
btnLdDat = tk.Button(place3, text="Load Data", bg="lightgray", command=nnLoadData)
btnLdDat.pack(side=tk.LEFT)
btnSvXorDat = tk.Button(
    place3, text="Sav Xor Dt", bg="lightgray", command=nnSaveXorData
)
btnSvXorDat.pack(side=tk.LEFT)
btnExit = tk.Button(place3, text="Exit2", width=5, bg="lightgray", command=quitcmd)
btnExit.pack(side=tk.RIGHT, padx=5)
# .............................................
# btnExit.place(x=wdt-100)
app.mainloop()
# .............................................
