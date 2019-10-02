import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
import random

'''
This file does not generate random trajectories. Instead, it takes the exact measured positions on the hall of St.
Petersburger Strasse and fills the testing and validation files with them to be able to check the performance of the system
with inaccurate measurements.

'''

real = False
collision_time = False
ncolision = 0
test = True

'''
The file to be written is chosen
'''
for n in range(6):
    if n<5:
        name =  "data/train/tudresden/residence_real" + str(n+1) + ".txt"
    else:
        name = "data/validation/tudresden/residence_real.txt"
    ftr = open(name, 'w')
    fte = open("data/test/tudresden/residence_real.txt", 'w')
    devx = 0


    ini = 0
    id = 0
    '''
    Access to the file where the coordinates are written is granted
    '''
    filename = "traj1.txt"
    f = open(filename, 'r')
    traj1 = f.read().split('\n')
    filename = "traj2.txt"
    f = open(filename, 'r')
    traj2 = f.read().split('\n')
    filename = "traj3.txt"
    f = open(filename, 'r')
    traj3 = f.read().split('\n')
    filename = "traj4.txt"
    f = open(filename, 'r')
    traj4 = f.read().split('\n')

    '''
    200 trajectories are written in a random order, controlled by k.
    '''
    for i in range(200):
        id = id + random.randint(1, 10)
        k = random.randint(1, 4)
        ini = ini + 200 + random.randint(1, 50) * 10

        if k == 1:
            ''' 
            If the trajectory is of type 1
            '''
            for j in range(20):
                '''
                The frame and the ID of the pedestrian are written
                '''
                ftr.write(str(ini + j * 10))
                ftr.write(" ")
                ftr.write(str(id))
                ftr.write(" ")
                '''
                The coordinates are written
                '''
                ftr.write(traj1[j].split(" ")[1])
                ftr.write(" ")
                ftr.write(traj1[j].split(" ")[0])
                ftr.write("\n")

                if j < 8:
                    '''
                    GROUND TRUTH
                    '''
                    fte.write(str(ini + j * 10))
                    fte.write(" ")
                    fte.write(str(id))
                    fte.write(" ")
                    fte.write(traj1[j].split(" ")[1])
                    fte.write(" ")
                    fte.write(traj1[j].split(" ")[0])
                    fte.write("\n")
                else:
                    '''
                    PART OF THE TRAJECTORY TO BE PREDICTED
                    '''
                    fte.write(str(ini + j * 10))
                    fte.write(" ")
                    fte.write(str(id))
                    fte.write(" ")
                    fte.write("?")
                    fte.write(" ")
                    fte.write("?")
                    fte.write("\n")

        if k == 2:
            for j in range(20):
                ftr.write(str(ini + j * 10))
                ftr.write(" ")
                ftr.write(str(id))
                ftr.write(" ")
                ftr.write(traj2[j].split(" ")[1])
                ftr.write(" ")
                ftr.write(traj2[j].split(" ")[0])
                ftr.write("\n")

                if j < 8:
                    fte.write(str(ini + j * 10))
                    fte.write(" ")
                    fte.write(str(id))
                    fte.write(" ")
                    fte.write(traj2[j].split(" ")[1])
                    fte.write(" ")
                    fte.write(traj2[j].split(" ")[0])
                    fte.write("\n")
                else:
                    fte.write(str(ini + j * 10))
                    fte.write(" ")
                    fte.write(str(id))
                    fte.write(" ")
                    fte.write("?")
                    fte.write(" ")
                    fte.write("?")
                    fte.write("\n")
        if k == 3:
            for j in range(20):
                ftr.write(str(ini + j * 10))
                ftr.write(" ")
                ftr.write(str(id))
                ftr.write(" ")
                ftr.write(traj3[j].split(" ")[1])
                ftr.write(" ")
                ftr.write(traj3[j].split(" ")[0])
                ftr.write("\n")

                if j < 8:
                    fte.write(str(ini + j * 10))
                    fte.write(" ")
                    fte.write(str(id))
                    fte.write(" ")
                    fte.write(traj3[j].split(" ")[1])
                    fte.write(" ")
                    fte.write(traj3[j].split(" ")[0])
                    fte.write("\n")
                else:
                    fte.write(str(ini + j * 10))
                    fte.write(" ")
                    fte.write(str(id))
                    fte.write(" ")
                    fte.write("?")
                    fte.write(" ")
                    fte.write("?")
                    fte.write("\n")

        if k == 4:
            for j in range(20):
                ftr.write(str(ini + j * 10))
                ftr.write(" ")
                ftr.write(str(id))
                ftr.write(" ")
                ftr.write(traj4[j].split(" ")[1])
                ftr.write(" ")
                ftr.write(traj4[j].split(" ")[0])
                ftr.write("\n")

                if j < 8:
                    fte.write(str(ini + j * 10))
                    fte.write(" ")
                    fte.write(str(id))
                    fte.write(" ")
                    fte.write(traj4[j].split(" ")[1])
                    fte.write(" ")
                    fte.write(traj4[j].split(" ")[0])
                    fte.write("\n")
                else:
                    fte.write(str(ini + j * 10))
                    fte.write(" ")
                    fte.write(str(id))
                    fte.write(" ")
                    fte.write("?")
                    fte.write(" ")
                    fte.write("?")
                    fte.write("\n")

    fte.close()
    ftr.close()




