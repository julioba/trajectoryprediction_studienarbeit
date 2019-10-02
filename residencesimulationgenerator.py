import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
import random



'''
This file generates trajectories for the scenario STPET_RESIDENCE
'''

'''
IMPORTANT PARAMETERS:

a:           Delay of the first step of the pedestrian
boundary:    Total length of the hall that pedestrians can not exceed
devy:        y deviation of the trajectory
devx:        x deviation of the trajectory
g:           Safety margin that the pedestrian lets behind it and the limit 
             of the obstacles
id:          ID. of the pedestrian
ini:         Frame number in which the pedestrian enters
k:           Identificator of the trajectory
osc_level:   Maximum level of oscillation of a pedestrian. It is 
             modelled as a random uniform function.
s:           Speed scaling factor

Data:        Ground truth of the trajectory
Results:     Steps that must be predicted


'''

'''
The general variables are defined
'''
ini = 0
next_ini = 0
k = 0
next_k = 0
id = 0
boundary = 7.8 # total length of the hall
g = 0.1 # Safety distance to the objects in m
maxdifx = 0.5 - g  # Maximum x distance to the closest object
maxdify = 0.5 - g  # Maximum y distance to the closest object

ncolision = 0 # No interactions are counted at the beginning
real = False
speed_max = 1.35
collision_time = False # No collision is supposed to happen now

'''
The 7 files (1 test, 1 validation and 5 training) are written as follows:
'''
for n in range(6):
    name = "data/train/stpet_residence/residence_training_" + str(n+1) + ".txt"
    if (n<5):
        f1 = open(name, 'w')
    else:
        f1 = open("data/validation/stpet_residence/stpet_residence_validation.txt", 'w')

    f2 = open("data/test/stpet_residence/residence_simulation.txt", 'w')
    '''
    200 trajectories are created:
    '''
    for i in range(200):
        def trajectory(k, s):
            Trajectory = [[0, 0] for i in range(20)]
            a = random.uniform(0,1)
            # FIRST TYPE OF TRAJECTORY
            if k == 0:
                last = 0
                for j in range(20):
                    if last > -boundary/2:
                        '''
                        The x variable is created uing the iteration j, the speed factor s, the first 
                        sampling moment a and the scaling factor (constant for each traj.)
                        '''
                        Trajectory[j][0] = (
                                    -0.0001 * (0.95*s*(j+a)) ** 4 + 0.004 * (0.95*s*(j+a)) ** 3 - 0.0492 * (0.95*s*(j+a)) ** 2 + 0.5211 * (0.95*s*(j+a))) - 4.0235 + 0.5
                        x = Trajectory[j][0]
                        '''
                        The y variable is created as a function of the x
                        '''
                        Trajectory[j][
                            1] = -0.0037 * x ** 6 - 0.0276 * x ** 5 - 0.0715 * x ** 4 + 0.0054 * x ** 3 + 0.2667 * x ** 2 - 1.0191 * x - 0.631
                        last = Trajectory[j][1]

                    else:
                        '''
                        If the point is generated out of the boundaries of the hall, it is not observed
                        '''
                        Trajectory[j][0] = Trajectory[j - 1][0]
                        Trajectory[j][1] = Trajectory[j - 1][1]

                for j in range(20):
                    '''
                    Possible mistakes are corrected
                    '''
                    if Trajectory[j][1] < -boundary/2:
                        Trajectory[j][0] = Trajectory[j - 1][0]
                        Trajectory[j][1] = Trajectory[j - 1][1]


            # SECOND TYPE OF TRAJECTORY
            '''
            The structure is the same as above
            '''
            if k == 1:
                last = 0
                for j in range(20):
                    if last > -boundary/2:
                        Trajectory[j][0] =  (
                                    0.0001 * (s * 1.2 *(j +a)) ** 4 - 0.0021 * (s * 1.2 *(j+a)) ** 3 - 0.0094 * (s * 1.2 *(j+a)) ** 2 - 0.0756 * (s * 1.2 *(j+a))) + 2.0891
                        x = Trajectory[j][0]
                        Trajectory[j][1] = -0.0149 * x ** 3 - 0.1487 * x ** 2 - 0.5982 * x - 2.0355
                        last = Trajectory[j][0]
                    else:
                        Trajectory[j][0] = Trajectory[j - 1][0]
                        Trajectory[j][1] = Trajectory[j - 1][1]
                Trajectory1 = Trajectory
                for j in range(20):
                    if Trajectory1[j][0] < -boundary/2:
                        Trajectory[j][0] = Trajectory[j - 1][0]
                        Trajectory[j][1] = Trajectory[j - 1][1]

            # THIRD TYPE OF TRAJECTORY
            '''
            The structure is the same as above
            '''

            if k == 2:
                last = 0
                for j in range(20):
                    if last > -boundary/2:
                        Trajectory[j][0] = (-0.0167 * (s * 1.2 * (j+a)) ** 2 + 0.0381 * (s * 1.2 *(j+a))) + 2.006 + 0.3
                        x = Trajectory[j][0]
                        Trajectory[j][1] = -0.0506 * x ** 3 - 0.324 * x ** 2 - 0.8222 * x + 0.0528
                        last = Trajectory[j][0]

                    else:
                        Trajectory[j][0] = Trajectory[j - 1][0]
                        Trajectory[j][1] = Trajectory[j - 1][1]

                for j in range(20):
                    if Trajectory[j][0] < -boundary/2:
                        Trajectory[j][0] = Trajectory[j - 1][0]
                        Trajectory[j][1] = Trajectory[j - 1][1]

            # FOURTH TYPE OF TRAJECTORY
            '''
            The structure is the same as above
            '''
            if k == 3:
                last = 0
                for j in range(20):
                    if last > -boundary/2:
                        Trajectory[j][0] =  (-0.0008 *(s *(j+a)) ** 3 + 0.0124 * (s *(j+a)) ** 2 + 0.4813 * (s *(j+a))) - 4.664 + 1.5
                        x = Trajectory[j][0]
                        Trajectory[j][1] = -0.0181 * x ** 3 - 0.1104 * x ** 2 - 0.1668 * x - 3.0263
                        last = Trajectory[j][1]
                    else:
                        Trajectory[j][0] = Trajectory[j - 1][0]
                        Trajectory[j][1] = Trajectory[j - 1][1]

            return Trajectory


        speed_i = np.random.normal(1.2, 0.2)
        s = speed_i / speed_max #Scalating speed factor

        if i != 0:
            ini = next_ini  # random.randint(0,40)*10
        k = next_k
        next_k = random.randint(0,3)
        next_ini = ini + random.randint(20, 40) * 10
        collision_time = False

        id += random.randint(1, 10) # The identity is randomly updated


        devy = random.uniform(-maxdify, maxdify)  # y deviation of the trajectory
        devx = random.uniform(-maxdifx, maxdifx)  # x deviation of the trajectory

        osc_level = random.uniform(0, g)  # Maximum level of oscillation of a pedestrian

        Trajectory = trajectory(k, s)
        trajectory = np.array(Trajectory, dtype='float')
        Data = Trajectory[0:8]
        data = np.array(Data, dtype='float')
        Results = Trajectory[8:20]
        results = np.array(Results, dtype='float')
        x0, y0 = data.T
        x1, y1 = results.T


        for j in range(20):
            if j < 8:
                '''
                Ground truth
                '''
                if real == False:
                    resy = trajectory[j][1] + devy + random.uniform(-osc_level,osc_level) # Deviation and oscillation are added
                    resx = trajectory[j][0] + devx + random.uniform(-osc_level,osc_level) # Deviation and oscillation are added
                    # frame
                    f1.write(str(ini + j * 10))
                    f1.write(" ")
                    # ID
                    f1.write(str(id))
                    f1.write(" ")
                    # y position
                    f1.write(str(resy))
                    f1.write(" ")
                    # x position
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
                    '''
                    Trajectory to be predicted
                    '''
                    resy = trajectory[j][1] + devy +  random.uniform(-osc_level,osc_level) # Deviation and oscillation are added
                    resx = trajectory[j][0] + devx + random.uniform(-osc_level,osc_level) # Deviation and oscillation are added
                    # frame
                    f1.write(str((j) * 10 + ini))
                    f1.write(" ")
                    # id
                    f1.write(str(id))
                    f1.write(" ")
                    # y position
                    f1.write(str(resy))
                    f1.write(" ")
                    # x position
                    f1.write(str(resx))
                    f1.write("\n")
                    f2.write(str((j) * 10 + ini))
                    f2.write(" ")
                    f2.write(str(id))
                    f2.write(" ")
                    # in the test files, the 12 last positions are unknwon
                    f2.write("?")
                    f2.write(" ")
                    f2.write("?")
                    f2.write("\n")

    f1.close()
    f2.close()




