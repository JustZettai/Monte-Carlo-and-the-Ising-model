import matplotlib
matplotlib.use('TKAgg')

import sys
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
import time

st = time.time()

#input
if(len(sys.argv) != 5):
    print ("Usage python Ising.py N T M A")
    sys.exit()

lx=int(sys.argv[1]) 
ly=lx 
kT=float(sys.argv[2]) 
model = str(sys.argv[3])
anim = str(sys.argv[4])

J=1.0
nStep = 5000000
measurementStep = 5000
kTStep = 0.1
equilibrium = 10000
tRange = np.arange((kT -kTStep), (kT + kTStep + 2), kTStep)
energyArr = np.zeros(int(nStep/measurementStep))
magnetisationArr = np.zeros(int(nStep/measurementStep))

averageE = []
averageM = []
capacity = []
susceptibility = []
uncertaintyEnergy = []
uncertaintyHeat = []
uncertaintyMag = []
uncertaintySus = []


spin=np.zeros((lx,ly),dtype=float)

#initialise spins randomly
for i in range(lx):
    for j in range(ly):
        r=random.random()
        if(r<0.5): spin[i,j]=-1
        if(r>=0.5): spin[i,j]=1

#compute delta E for Glauber
def deltaEG(i, j):

    neighbourSum = 0
    neighbourSum += spin[i%lx,(j+1)%lx]
    neighbourSum += spin[i%lx,(j-1)%lx]
    neighbourSum += spin[(i+1)%lx,j%lx]
    neighbourSum += spin[(i-1)%lx,j%lx]
    return 2*spin[i,j]*neighbourSum

#compute delta E for Kawasaki by two spin changes and correction
def deltaEK(i1, j1, i2, j2, check):
# pass as an argument how the neighbours are positioned and appropriately calculate the double spin change
    if (check == 0):
        neighbourSumOne = 0
        neighbourSumTwo = 0

        neighbourSumOne += spin[i1%lx,(j1+1)%lx]
        neighbourSumOne += spin[i1%lx,(j1-1)%lx]
        neighbourSumOne += spin[(i1+1)%lx,j1%lx]
        neighbourSumOne += spin[(i1-1)%lx,j1%lx]
        
        neighbourSumTwo += spin[i2%lx,(j2-1)%lx]
        neighbourSumTwo += spin[i2%lx,(j2+1)%lx]
        neighbourSumTwo += spin[(i2+1)%lx,j2%lx]
        neighbourSumTwo += spin[(i2-1)%lx,j2%lx]

        return 2*spin[i1,j1]*neighbourSumOne + 2*spin[i2,j2]*neighbourSumTwo
    elif (check == 1 ):
        neighbourSumOne = 0
        neighbourSumTwo = 0

        neighbourSumOne += spin[i1%lx,(j1-1)%lx]
        neighbourSumOne += spin[(i1+1)%lx,j1%lx]
        neighbourSumOne += spin[(i1-1)%lx,j1%lx]
        
        neighbourSumTwo += spin[i2%lx,(j2+1)%lx]
        neighbourSumTwo += spin[(i2+1)%lx,j2%lx]
        neighbourSumTwo += spin[(i2-1)%lx,j2%lx]

        return 2*spin[i1,j1]*neighbourSumOne + 2*spin[i2,j2]*neighbourSumTwo
    elif (check == 2 ):
        neighbourSumOne = 0
        neighbourSumTwo = 0

        neighbourSumOne += spin[i1%lx,(j1+1)%lx]
        neighbourSumOne += spin[(i1+1)%lx,j1%lx]
        neighbourSumOne += spin[(i1-1)%lx,j1%lx]
        
        neighbourSumTwo += spin[i2%lx,(j2-1)%lx]
        neighbourSumTwo += spin[(i2+1)%lx,j2%lx]
        neighbourSumTwo += spin[(i2-1)%lx,j2%lx]

        return 2*spin[i1,j1]*neighbourSumOne + 2*spin[i2,j2]*neighbourSumTwo
    elif (check == 3 ):
        neighbourSumOne = 0
        neighbourSumTwo = 0

        neighbourSumOne += spin[i1%lx,(j1-1)%lx]
        neighbourSumOne += spin[i1%lx,(j1+1)%lx]
        neighbourSumOne += spin[(i1-1)%lx,j1%lx]
        
        neighbourSumTwo += spin[i2%lx,(j2+1)%lx]
        neighbourSumTwo += spin[i2%lx,(j2-1)%lx]
        neighbourSumTwo += spin[(i2+1)%lx,j2%lx]

        return 2*spin[i1,j1]*neighbourSumOne + 2*spin[i2,j2]*neighbourSumTwo
    elif (check == 4 ):
        neighbourSumOne = 0
        neighbourSumTwo = 0

        neighbourSumOne += spin[i1%lx,(j1-1)%lx]
        neighbourSumOne += spin[i1%lx,(j1+1)%lx]
        neighbourSumOne += spin[(i1+1)%lx,j1%lx]
        
        neighbourSumTwo += spin[i2%lx,(j2+1)%lx]
        neighbourSumTwo += spin[i2%lx,(j2-1)%lx]
        neighbourSumTwo += spin[(i2-1)%lx,j2%lx]

        return 2*spin[i1,j1]*neighbourSumOne + 2*spin[i2,j2]*neighbourSumTwo
    

#calculate error bars using a bootsrap method
def errorBars(arr, t, val):
    reSamples = 100
    errorBar = np.zeros(reSamples)
    size = len(arr)
    sample = np.zeros(size)
    if (val == "c"):
        for e in range (0, len(errorBar)):
            for s in range (0, size):
                sample[s] = arr[random.randint(0, size-1)]
            errorBar[e] = (1/(lx*ly*t**2)*(np.mean(sample**2) - np.mean(sample)**2))
    elif (val == "s"):
        for e in range (0, len(errorBar)):
            for s in range (0, size):
                sample[s] = arr[random.randint(0, size-1)]
            errorBar[e] = (1/(lx*ly*t)*(np.mean(sample**2) - np.mean(sample)**2))
    return np.sqrt(np.mean(errorBar**2)- np.mean(errorBar)**2)

def standandError(foo):
    return np.sqrt((np.mean(foo**2)-np.mean(foo)**2)/(nStep/measurementStep))

def measurement(n):
#take measurements as sppecified a the beginning of the code
    if(n%measurementStep == 0): 
#       update measurements
        energy = 0
        magnetisation = 0
        for i in range(lx):
            for j in range(ly):
                energy +=  -spin[i,j]*(spin[(i+1)%lx, j] + spin[i, (j+1)%ly])
                magnetisation += spin[i,j]
                
        energyArr[int((n-equilibrium)/measurementStep)] = energy
        magnetisationArr[int((n-equilibrium)/measurementStep)] = magnetisation
#       show animation
        if (anim.lower() == "y"):
            plt.cla()
            plt.imshow(spin, animated=True)
            plt.draw()
            plt.pause(0.00001)

hour = datetime.datetime.now().strftime("%H_%M")

#open files to save measurments
f = open('measurment_%s.dat' % (hour),'w')
c = open('heat_capacity%s.dat' % (hour),'w')
for t in tRange:
#   update loop depending on the model used
    if (model.lower() == "g"):
        for n in range(nStep + equilibrium):
#       select spin randomly
            itrial=np.random.randint(0,lx)
            jtrial=np.random.randint(0,ly)
            spin_new=-spin[itrial,jtrial]
            de = deltaEG(itrial,jtrial)

#           perform metropolis test
            r = random.random()
            if(r <= np.exp(-de/t)):
                spin[itrial, jtrial] = spin_new
            
            if (n >= equilibrium):
                measurement(n)

#       append the neccesary list to measure the average values and uncertainties
        aEnergy = np.mean(energyArr)
        aMagnet = np.mean(abs(magnetisationArr))
        heatC = (1/(lx*ly*t**2)*(np.mean(energyArr**2) - np.mean(energyArr)**2))
        sus = (1/(lx*ly*t)*(np.mean(magnetisationArr**2) - np.mean(magnetisationArr)**2))
        stE = standandError(energyArr)
        stM = standandError(magnetisationArr)
        errorHeat = errorBars(energyArr, t, "c")
        errorSus = errorBars(magnetisationArr, t, "s")

        f.write('%s %s %s %s %s\n' % (round(t,2), aEnergy, heatC, aMagnet, sus))
        c.write("%s %s %s\n" % (round(t,2), heatC, errorHeat))

        averageE.append(aEnergy)
        uncertaintyEnergy.append(stE)
        averageM.append(aMagnet)
        uncertaintyMag.append(stM)
        capacity.append(heatC)
        uncertaintyHeat.append(errorHeat)
        susceptibility.append(sus)
        uncertaintySus.append(errorSus)

    elif (model.lower() == "k"):
        for n in range(nStep):
            check = 0
            i1=np.random.randint(0,lx)
            j1=np.random.randint(0,ly)
            i2=np.random.randint(0,lx)
            j2=np.random.randint(0,ly)

#           check for nearest neighbours
            if (spin[i1%lx, j1%ly] == spin[i2%lx, j2%ly]):
                changeE = 0
            elif (i1%lx == i2%lx and (j1 + 1)%lx == j2%lx):
                check = 1
                changeE = deltaEK(i1, j1, i2, j2, check)
            elif (i1%lx == i2%lx and (j1 - 1)%lx == j2%lx):
                check = 2
                changeE = deltaEK(i1, j1, i2, j2, check)
            elif ((i1 + 1)%lx == i2%lx and j1%lx == j2%lx):
                check = 3
                changeE = deltaEK(i1, j1, i2, j2, check)
            elif((i1 - 1)%lx == i2%lx and j1%lx == j2%lx):
                check = 4
                changeE = deltaEK(i1, j1, i2, j2, check)
            else:
                changeE = deltaEK(i1, j1, i2, j2, check)

#           perform metropolis test
            r = random.random()
            if(r <= np.exp(-changeE/t)):
                    temp = spin[i1, j1]
                    spin[i1, j1] =  spin[i2, j2]
                    spin[i2, j2] =  temp
            
            if (n >= equilibrium):
                measurement(n)

#       append the neccesary list to measure the average values and uncertainties
        aEnergy = np.mean(energyArr)
        heatC = (1/(lx*ly*t**2)*(np.mean(energyArr**2) - np.mean(energyArr)**2))
        stE = standandError(energyArr)
        errorHeat = errorBars(energyArr, t, "c")

        f.write('%s %s %s\n' % (round(t,2), aEnergy, heatC))
        c.write('%s %s %s\n' % (round(t,2), heatC, errorHeat))

        
        averageE.append(aEnergy)
        uncertaintyEnergy.append(stE)
        capacity.append(heatC)
        uncertaintyHeat.append(errorHeat)

#close datafiles with measurments 
f.close()
c.close()

#plot and save graphs depending on model used
if (model.lower() == "g"):
    plt.title("Energy of the system modeled via Glauber dynamics")
    plt.scatter(tRange[1:], averageE[1:], uncertaintyEnergy[1:])
    plt.errorbar(tRange[1:], averageE[1:], uncertaintyEnergy[1:], fmt="none", capsize = 3)
    plt.savefig("Energy_%s" % (hour))
    plt.close()

    plt.title("Heat capacity of the system modeled via Glauber dynamics")
    plt.scatter(tRange[1:], capacity[1:], uncertaintyHeat[1:])
    plt.errorbar(tRange[1:], capacity[1:], uncertaintyHeat[1:], fmt="none", capsize = 3)
    plt.savefig("Heat_%s" % (hour))
    plt.close()

    plt.title("Magnetisation of the system modeled via Glauber dynamics")
    plt.scatter(tRange[1:], averageM[1:], uncertaintyMag[1:])
    plt.errorbar(tRange[1:], averageM[1:], uncertaintyMag[1:], fmt="none", capsize = 3)
    plt.savefig("Mag_%s" % (hour))
    plt.close()

    plt.title("Susceptibility of the system modeled via Glauber dynamics")
    plt.scatter(tRange[1:], susceptibility[1:], uncertaintySus[1:])
    plt.errorbar(tRange[1:], susceptibility[1:], uncertaintySus[1:], fmt="none", capsize = 3)
    plt.savefig("Sus_%s" % (hour))
    plt.close()
    
elif (model.lower() == "k"):
    plt.title("Energy of the system modeled via Kawasaski dynamics")
    plt.scatter(tRange[1:], averageE[1:], uncertaintyEnergy[1:])
    plt.errorbar(tRange[1:], averageE[1:], uncertaintyEnergy[1:], fmt="none", capsize = 3)
    plt.savefig("Energy_%s" % (hour))
    plt.close()

    plt.title("Heat capacity of the system modeled via Kawasaki dynamics")
    plt.scatter(tRange[1:], capacity[1:], uncertaintyHeat[1:])
    plt.errorbar(tRange[1:], capacity[1:], uncertaintyHeat[1:], fmt="none", capsize = 3)
    plt.savefig("Heat_%s" % (hour))
    plt.close()

end = time.time()
print("Execution time: ", end-st)
