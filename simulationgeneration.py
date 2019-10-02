import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
import random


'''
This file generates the trajectories in the scenario of SIMULATION.
'''

'''
-------------------------------------------------------------------
                   DEFINITION OF THE CORE TRAJECTORIES
-------------------------------------------------------------------
'''
a = random.uniform(0,1)
ini = 0
next_ini = 0
k = 0
next_k = 0
id = 0
collision_time = False
ncolision = 0
test = True
'''
Trajectory 1 without interaction
'''
Data1 = [[2.5,-5 + j*((5 - 0.5)/8)]for j in range(8)]
data1 = np.array(Data1, dtype = 'float')
Results1 = [[0.5+ 2*np.cos(j/6),-0.5 + j*(5.5/12)] for j in range(12)]
results1 = np.array(Results1, dtype = 'float')
x1, y1 = data1.T
xx1,yy1 = results1.T
plt.scatter(x1,y1, color = 'g')
plt.scatter(xx1,yy1)
plt.grid(True)
plt.ylim((-5,5))
plt.xlim((-5,5))

'''
Trajectory 2 without interaction
'''
Data2 = [[(20-j-10)/2,((5-2.5/7.5*(j+a))-5)]for j in range(8)]
data2 = np.array(Data2, dtype = 'float')
Results2 = [[1-(j+a-8)*6.5/12, ((10 - (5 + 3*np.sin(1 +1*np.pi/360*10*(20-j-a))))-5)] for j in range(8,20)]
results2 = np.array(Results2, dtype = 'float')
x2, y2 = data2.T
xx2,yy2 = results2.T
plt.scatter(x2,y2)
plt.scatter(xx2,yy2)
plt.grid(True)
plt.ylim((-5,5))
plt.xlim((-5,5))

'''
Trajectory 3
'''
Data3 = [[ (j/20*17 - 10)/2, ((+3-j*17/20*3/17)-5)]for j in range(8)]
data3= np.array(Data3, dtype = 'float')
Results3 = [[((j/20*17)-10)/2,((3-j/20*17*3/17)-5)] for j in range(8,20)]
results3 = np.array(Results3, dtype = 'float')
x3, y3 = data3.T
xx3,yy3 = results3.T
plt.scatter(x3,y3)
plt.scatter(xx3,yy3)
plt.grid(True)
plt.ylim((-5,5))
plt.xlim((-5,5))

'''
Trajectory 4
'''
Data4 = [[(10-j)/2, ((5.5 + 3*np.sin(2*np.pi/36*(20-j)))-5)]for j in range(8)]
data4 = np.array(Data4, dtype = 'float')
Results4 = [[((20-j/20*20)-10)/2, ((7.5- np.cos(2*np.pi/20*(20-j)))-5)] for j in range(8,20)]
results4 = np.array(Results4, dtype = 'float')
x4, y4 = data4.T
xx4,yy4 = results4.T

'''
Trajectory 1 with interaction with Traj. 4
'''
Data1_a4 = [[2.5 + np.sin(j/4), -5 + j*4.5/8]for j in range(8)]
data1_a4 = np.array(Data1_a4, dtype = 'float')
Results1_a4 = [[1.5+ 2*np.cos(j/4),-0.5 + j*(5.5/12)] for j in range(12)]
results1_a4 = np.array(Results1_a4, dtype = 'float')
x1a4, y1a4 = data1_a4.T
xx1a4,yy1a4 = results1_a4.T
plt.scatter(x1a4,y1a4, color = 'g')
plt.scatter(xx1a4,yy1a4)
plt.grid(True)
plt.ylim((-5,5))
plt.xlim((-5,5))
'''
Trajectory 1 with interaction with Traj. 2
'''
Data1_a2 = [[2.5,-5 + j*((5 - 0.5)/8)]for j in range(8)]
data1_a2 = np.array(Data1_a2, dtype = 'float')
Results1_a2 =  [[0.5+ 2*np.cos(j/3.5)+ 0.0001*j**4,-0.5 + j*(5.5/12)] for j in range(12)]
results1_a2 = np.array(Results1_a2, dtype = 'float')
x1a2, y1a2 = data1.T
xx1a2,yy1a2 = results1_a2.T
plt.scatter(x1a2,y1a2, color = 'g')
plt.scatter(xx1a2,yy1a2, color = 'r')
plt.grid(True)
plt.ylim((-5,5))
plt.xlim((-5,5))


'''
Trajectory 2 interacting with Trajectory 4
'''
Data2_a4 = [[(20-j-10)/2,((5-1.5/7.5*(j))-5)]for j in range(8)]
data2_a4 = np.array(Data2_a4, dtype = 'float')
Results2_a4 = [[1-(j-8)*6.5/12, ((10 - (4 + 3*np.sin(1 +1*np.pi/360*10*(20-j))))-5)] for j in range(8,20)]
results2_a4 = np.array(Results2_a4, dtype = 'float')
x2a4, y2a4 = data2_a4.T
xx2a4,yy2a4 = results2_a4.T
plt.scatter(x2a4,y2a4)
plt.scatter(xx2a4,yy2a4)
plt.grid(True)
plt.ylim((-5,5))
plt.xlim((-5,5))


'''
Trajectory 3 interacting with trajectory 1
'''
Data3_a1 = [[ (j/20*17 - 10)/2, ((+3-j*17/20*3/17)-5)]for j in range(8)]
data3_a1= np.array(Data3_a1, dtype = 'float')
Results3_a1 = [[-0.5 + (j-12)*3.5/12,((3-j/20*12*3/17)-5)-0.5] for j in range(8,20)]
results3_a1 = np.array(Results3_a1, dtype = 'float')
x3a1, y3a1 = data3_a1.T
xx3a1,yy3a1 = results3_a1.T
plt.scatter(x3a1,y3a1)
plt.scatter(xx3a1,yy3a1)
plt.grid(True)
plt.ylim((-5,5))
plt.xlim((-5,5))

'''
Trajectory 3 interacting with trajectory 2
'''
Data3_a3 = [[ -5 + j*3/8, -2 - 2*np.sin(j/4.5)]for j in range(8)]
data3_a3= np.array(Data3_a3, dtype = 'float')
Results3_a3 = [[-2+(j-8)*6/12,-4 - (-8+j)/12] for j in range(8,20)]
results3_a3 = np.array(Results3_a3, dtype = 'float')
x3a3, y3a3 = data3_a3.T
xx3a3,yy3a3 = results3_a3.T
plt.scatter(x3a3,y3a3)
plt.scatter(xx3a3,yy3a3)
plt.grid(True)
plt.ylim((-5,5))
plt.xlim((-5,5))

'''
Trajectory 4 interacting with Trajectory 1
'''
Data4_a1 = [[5- j*4/8,-0.5 + 5*np.sin(j/5)]for j in range(8)]
data4_a1= np.array(Data4_a1, dtype = 'float')
Results4_a1 = [[0.9 - (j-8)*6/12,-0.5 + 5*np.sin(j/11+np.pi/4) ] for j in range(8,20)]
results4_a1 = np.array(Results4_a1, dtype = 'float')
x4a1, y4a1 = data4_a1.T
xx4a1,yy4a1 = results4_a1.T


plt.scatter(x1,y1, color ='r', label = 'Traj_1')
plt.scatter(xx1,yy1,  color ='r')
plt.scatter(x4,y4,color ='y', label = 'Traj_2')
plt.scatter(xx4,yy4,color ='y')
plt.scatter(x2,y2,color ='g', label = 'Traj_3')
plt.scatter(xx2,yy2,color ='g')
plt.scatter(x3,y3,  color ='b', label = 'Traj_4')
plt.scatter(xx3,yy3, color = 'b')
plt.legend(loc='best')

plt.grid(True)
plt.ylim((-5,5))
plt.xlim((-5,5))
plt.axis('equal')

plt.savefig('true_traj.eps')


'''
VECTORS CONTAINING ALL THE DATA AND ALL THE RESULTS
'''
data = [data1, data2, data3, data4, data1_a2, data1_a4, data2_a4, data3_a1, data3_a3, data4_a1]
results = [results1, results2, results3, results4, results1_a2, results1_a4, results2_a4, results3_a1, results3_a3,
           results4_a1]


'''
----------------------------------------------------------------------
                    GENERATION OF THE DATASETS
----------------------------------------------------------------------
'''
'''
5 files with 200 trajectories each are created in the following, one in each iteration. They are saved in
simulationtraining1.txt, simulationtraining2.txt, simulationtraining3.txt, simulationtraining4.txt
and simulationtraining5.txt.

Then, a last set of 200 trajectories is saved in simulation_test and simulation_validation. Note that the 
trajectory is the same, but in simulation_test, the 12 last points of each trajectory are occupied by '?' instead 
of the real coordinates.
'''

interaction_threshold = 1000 # If interaction wants to be modelled, the threshold is to be set lower. I.e, if the threshold
# is 0, an interaction is modelled for each iteration
for s in range(6):
    datasetname = "data/train/simulation/simulationtraining" + str(s+1) + ".txt"
    f2 = open("data/test/simulation/simulation_test.txt", 'w')
    if s < 5:
        f1 = open(datasetname, 'w')
        print(datasetname)

    else:
        f1 = open("data/validation/simulation/simulation_validation.txt", 'w')
    for i in range(200):
        g = 0.2 # Safety distance to the objects in m
        maxdifx = 0.5 - g# Maximum x distance to the closest object
        maxdify = 0.5 - g # Maximum y distance to the closest object
        devy = random.uniform(-maxdify, maxdify) # y deviation of the trajectory
        devx = random.uniform(-maxdifx, maxdifx) # x deviation of the trajectory
        osc_level = random.uniform(0,g) # Maximum level of oscillation of a pedestrian

        if collision_time == False:
            collision = random.randint(0, 100)
        else:
            collision = 0

        if collision < interaction_threshold:  # If no interaction occurs
            if i != 0:
                ini = next_ini # Where ini is the entry number of frame of the pedestrian
            k = next_k
            next_k = random.randint(0, 3)
            next_ini = ini + random.randint(0, 40) * 10
            collision_time = False


        else:  # If a collision occurs

            collision_time = True
            if collision > 0:  # collision 1-2
                ini = next_ini
                k = 4
                next_k = 9
                next_ini = ini + 10
            else:
                if collision > 60:  # collision 1 o 4
                    ini = next_ini
                    k = 6
                    next_k = 8
                    next_ini = ini + 80
                else:  # collision 3 o 4
                    ini = next_ini
                    k = 5
                    next_k = 7
                    next_ini = ini + 80

        id += random.randint(1, 10)
        for j in range(20):
            if j < 8:
                resy = data[k][j][1] + devy + random.uniform(-osc_level,osc_level) # The deviation and the oscillation
                # are added
                resx = data[k][j][0] + devx + random.uniform(-osc_level,osc_level)
                f1.write(str(ini + j * 10))
                f1.write(" ")
                f1.write(str(id))
                f1.write(" ")
                f1.write(str(resy))
                f1.write(" ")
                f1.write(str(resx))
                f1.write("\n")
                f2.write(str(ini + j * 10))
                f2.write(" ")
                f2.write(str(id))
                f2.write(" ")
                f2.write(str(resy))
                f2.write(" ")
                f2.write(str(resx))
                f2.write("\n")
            else:
                resy = results[k][j - 8][1] + devy + random.uniform(-osc_level,osc_level)
                resx = results[k][j - 8][0] + devx + random.uniform(-osc_level,osc_level)
                f1.write(str((j) * 10 + ini))
                f1.write(" ")
                f1.write(str(id))
                f1.write(" ")
                f1.write(str(resy))
                f1.write(" ")
                f1.write(str(resx))
                f1.write("\n")
                f2.write(str((j) * 10 + ini))
                f2.write(" ")
                f2.write(str(id))
                f2.write(" ")
                f2.write("?")
                f2.write(" ")
                f2.write("?")
                f2.write("\n")

    f1.close()
    f2.close()
    ini = 0
    next_ini = 0
    k = 0
    next_k = 0






