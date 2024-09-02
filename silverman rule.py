# Monte Carlo Simulations

import scipy
from scipy.special import ndtr as N
from scipy.integrate import quad
from scipy.stats import qmc, norm, kurtosis, skew
from timeit import timeit
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures as cf
from numpy import log, pi
import pandas as pd




# Initial Parameters

BTC0 = np.log(1200)
rho = -np.sqrt(0.5)
drift = 0.0
k = 5
gamma = 0.0225
v = 0.4
jumpActivity = 480
jumpMean = 0
jumpStd = 1
jumpSize = 0.007
sig = 0.0225
newLim = 0.04**2
jumpLim = 9999
startTime = 6000

Taun = 126310.0
Tau = 126350.0

pnTheta = 0.45
gradJumpSize = 0.02

dbTheta = 0.65
c=3



pn = False
db = False

etaValues = []
for i in range(34,51):
    etaValues.append(i/10)

w = 30
r = 5

detectionsTotal = 0

stack = deque(maxlen=w)
bandwidth = 0.0108



###################################
#------------------------------------------------------------------------------------------#
###################################
test49 = []
test491 = []
test26 = []
test261 = []


# Define Batch Process for multistep QMC
def batchProcess(data, loop, batchSize, intervals, bandwidth):

    K = 480
    
    badCount = np.zeros(len(etaValues))
    detection = np.zeros((batchSize, len(etaValues)))
    falseDetection = np.zeros((batchSize, len(etaValues)))
    ARL = np.zeros((batchSize, len(etaValues)))
    detectionDelay = np.zeros((batchSize, len(etaValues)))

    dt = 1/(24*20*365)

    BTCVals = np.full((batchSize,intervals+1),BTC0)
    DeltaBTC = np.full((batchSize,intervals+1),0.0)
    sigVals = np.full((batchSize,intervals+1),sig)
    HVals = np.full((batchSize,intervals+1),0.0)

    


    np.random.seed(12345)
    N = np.random.poisson(jumpActivity * dt, (batchSize, intervals))
    J = np.random.normal(jumpMean, jumpStd, (batchSize, intervals))

    jumpsDetected = np.count_nonzero(N[:,K:])
    print('jump =', jumpsDetected)

    if db == False and pn == False:
        
        # Generate price paths
        for t in range(1,intervals+1):
            if t % 1000 == 0:
                print('t=',t)
            for i in range(batchSize):
                sigVals[i,t] = max(sigVals[i,t-1] + k*(gamma-sigVals[i,t-1])*dt + v*np.sqrt(sigVals[i,t-1])*np.random.normal(jumpMean, jumpStd)*np.sqrt(dt), 0)
                DeltaBTC[i,t-1] = drift*dt + np.sqrt(sigVals[i,t])*np.random.normal(jumpMean, jumpStd)*np.sqrt(dt) + N[i, t-1] * J[i, t-1] * jumpSize
                if i==16:
                    test261.append(abs(data[i+loop*batchSize,1]))
                    test26.append(DeltaBTC[i,t-1])

                    
                # if abs(N[i, t-1]) >0:
                #     print(N[i, t-1],N[i, t-1] * J[i, t-1] * jumpSize,i,t)




                # if abs(DeltaBTC[i,t-1])>jumpLim:
                #     DeltaBTC[i,t-1] = 0
                BTCVals[i,t] = BTCVals[i,t-1] + DeltaBTC[i,t-1]

                

            np.random.seed(t)
            np.random.shuffle(data)

    #jump removal

    n = 175200
    a = 0.1
    c = np.sqrt(2/np.pi)
    b = c*np.sqrt(2*np.log(n))
    an = np.sqrt(2*log(n))/c - (log(pi)+log(log(n)))/(2*b)
    print(a,b,c)

    for i in range(0,batchSize):
        # print(i)
        if i % 100 == 0:
            print('i=',i)
        for t in range(K,intervals+1):
            mu = 1/(K-1) * np.sum(DeltaBTC[i,t-K+1:t-1])
            if i==0 and t == K:
                print('mu =', mu)

            tempVolEst = np.sqrt(1/(K-2) * np.sum(abs(DeltaBTC[i,t-K:t-2]) * abs(DeltaBTC[i,t-K+1:t-1])))
            if i==0 and t == K:
                print('tempVolEst =', tempVolEst)

            T = (DeltaBTC[i,t-1]-mu)/tempVolEst
            if i==0 and t == K:
                print('T =', T)

            beta = -log(-log(1-a))
            if i==0 and t == K:
                print('beta =', beta)
            # print(b*(np.abs(T)-an) - beta)
            if b*(np.abs(T)-an) - beta > 0:
                # print('crit vals =', b*(np.abs(T)-an), beta, b*(np.abs(T)-an) - beta, i, t)
                if i == 0:
                    print('crit vals =', b*(np.abs(T)-an), beta, b*(np.abs(T)-an) - beta, i, t)
                # print(DeltaBTC[i,t-1])

                DeltaBTC[i,t-1] = 0

                # print(DeltaBTC[i,t-1])


    # Stopping Rule
    stopRule = []


    sig1 = []
    sig2 = []
    errors = []
    errors2 = []
    sigW = startTime*2
    print('sigW=', sigW, len(DeltaBTC[i,K:-1]))
    groupSize = sigW // n

    delta = 1/(24*20*365)
    n = 480
    T=delta*n
    print(T,n,delta)
    silvermanBandwidths = []

    for i in range(batchSize):
        data = DeltaBTC[i,K:-1]

        # Step 2: Estimate Pilot Bandwidth
        n = 480
        std_dev = np.std(data)
        silvermanBandwidth = 1.06 * std_dev * n ** (-1/5)
        print('silverman =', silvermanBandwidth)


        silvermanBandwidths.append(silvermanBandwidth)

    print('silverman',np.mean(silvermanBandwidths))


def MCSims(iterations, loopLength, intervals, seed, bandwidth):

    plotGraphs = True

    loops = int(iterations/loopLength)
    print('loops =',loops)

    quasiDimension = 2
    x = qmc.Halton(quasiDimension, scramble=True, seed=seed)
    data = norm.ppf(x.random(np.array(iterations), workers=-1))
    data[:,1] = rho*data[:,0] + np.sqrt(1-rho**2)*data[:,1]

    # Submit tasks for parallel processing
    with cf.ThreadPoolExecutor() as executor:
        batches = [executor.submit(batchProcess, data, loop, loopLength, intervals, bandwidth) for loop in range(loops)]


    out = []
    stopRule = []
    # Plot graph to show completed
    fig, ax = plt.subplots()
    face = plt.Circle((0.5, 0.5), 0.4, color='yellow', ec='black', lw=2)
    ax.add_patch(face)
    eyeleft = plt.Circle((0.35, 0.65), 0.05, color='black')
    eyeright = plt.Circle((0.65, 0.65), 0.05, color='black')
    ax.add_patch(eyeleft)
    ax.add_patch(eyeright)
    theta = np.linspace(0, np.pi, 100)
    x = 0.5 + 0.2 * np.cos(theta)
    y = 0.4 + 0.1 * np.sin(-theta)
    ax.plot(x, y, color='black', lw=2)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.show()

    return

MCSims(1000, 1000, 960, 12345, bandwidth)
