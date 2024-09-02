import numpy as np
from scipy.integrate import quad, dblquad
import matplotlib.pyplot as plt


# Exponential kernel function
def kernel(x):
    return 0.5 * np.exp(-np.abs(x))

# Estimate integrated quarticity (IQ)
def eatimateIQ(diffX, delta):
    print((3 * delta) ** -1, np.sum(diffX ** 4))
    print('iq=', (3 * delta) ** -1 * np.sum(diffX ** 4))
    return (3 * delta) ** -1 * np.sum(diffX ** 4)

def C(x,y):
    # print((np.abs(x)+np.abs(y)-np.abs(x-y)), (1/2)*(np.abs(x)+np.abs(y)-np.abs(x-y)))
    return (1/2)*(np.abs(x)+np.abs(y)-np.abs(x-y))


def integrand(x, y):
    # print('x,y =', x,y)
    # print('c=', C(x, y))
    return kernel(x) * kernel(y) * C(x, y)

# Left and right side kernel estimators
def leftsidekernelestimator(diffX, h, t, delta, n):

    tj = np.arange(t+1)

    numerator = np.sum(kernel((tj-1)*delta/h)/h * np.square(diffX[tj]))
    denominator = delta * np.sum(kernel((tj-1)*delta/h)/h)


    # print(h)
    # print(kernel(tj*delta/h)/h, np.square(diffX[tj]))
    # print(numerator, denominator)
    return numerator / denominator



def rightsidekernelestimator(diffX, h, t, delta, n):
    # print(n)
    tj = np.arange(t+1,n)
    # print(t+1,n)
    # print('end=', kernel((tj[0]-1)*delta/h)/h, np.square(diffX[tj[0]]), )
    numerator = np.sum(kernel((tj-1)*delta/h)/h * np.square(diffX[tj]))
    denominator = delta * np.sum(kernel((tj-1)*delta/h)/h)

    # print(kernel(tj*delta/h)/h, np.square(diffX[tj]))
    # print(numerator, denominator)
    return numerator / denominator



# TSRVV estimator
def eatimateIVV(diffX, h, k, delta, b, n):

    n = len(diffX)-1
    # print(n, 'n')
    IVV = 0
    for i in range(b, n - k - b):
        deltasigmasquaredk = rightsidekernelestimator(diffX, h, i + k, delta, n) - leftsidekernelestimator(diffX, h, i, delta, n)
        deltasigmasquared1 = rightsidekernelestimator(diffX, h, i + 1, delta, n) - leftsidekernelestimator(diffX, h, i, delta, n)
        IVV += (deltasigmasquaredk ** 2 - ((n - k + 1) / n) * deltasigmasquared1 ** 2) / k
    return IVV / (n - k - b)

# Function to estimate optimal bandwidth
def estimateOptimalBandwidth(X, T, delta, n, numiterations=3, b=3, k=3):
    print(n)
    k=int(n**(2/3))
    print('k=',k)
    n = len(X)
    print(n)
    doubleintegralresult, _ = dblquad(integrand, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)
    # Initial bandwidth guess
    singleintegralresult = quad(lambda x: kernel(x) ** 2, -np.inf, np.inf)[0]

    initialbandwidth = np.sqrt((2 * T * singleintegralresult )/ (n*doubleintegralresult))


    bandwidth = initialbandwidth
    print('bandwidth =', initialbandwidth)

    diffX = np.diff(X)
    # print(diffX)

    for i in range(numiterations):
        print(i)
        IQ = eatimateIQ(diffX, delta)
        print(IQ)
        IVV = eatimateIVV(diffX, bandwidth, k, delta, b, n)
        print(IVV)
        bandwidth = np.sqrt((2 * T * IQ * singleintegralresult) / (n * IVV * doubleintegralresult))
        print('latest bandwidth estimate =', bandwidth)

    return bandwidth

# Example usage

# Simulated or actual data
np.random.seed(42)
n = 480  # Number of data points
T = 1  # Total time
delta = T / n  # Time increment
X = np.cumsum(np.random.randn(n))  # Simulated Brownian motion

# plt.plot(X)
# plt.show()

# optimalBandwidth = eatimateoptimalBandwidth(X, T, delta)
# print("Optimal Bandwidth,300:", optimalBandwidth)


inputFile = 'BTC prices inSamp.csv'
with open(inputFile, 'r') as file:
    priceData = np.array([float(line.strip()) for line in file])

newData = []
for i in range(0, len(priceData) - 1, 3):
    newData.append(priceData[i])

newData = newData[:481]
BTCVals = np.log(newData)

delta = 1/(24*20*365)
n = len(BTCVals)
T=delta*n
print(T,n,delta)

optimalBandwidth = estimateOptimalBandwidth(BTCVals, T, delta, n)
print("Optimal Bandwidth,bitcoin:", optimalBandwidth, optimalBandwidth/delta)


#////////////////////////////////////////////////////////////////////////////#



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
jumpActivity = 3/5
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
bandwidth = 0.08059071580479968
bandwidth = 0.002848776629694912
# bandwidth = 0.00003234

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
                # if i == 0:
                #     print('crit vals =', b*(np.abs(T)-an), beta, b*(np.abs(T)-an) - beta, i, t)
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
    optimalBandwidths = []

    for i in range(batchSize):
        print(i)
        data = DeltaBTC[i,K:-1]

        delta = 1/(24*20*365)
        # n = len(BTCVals[0])
        T=delta*n
        print(T,n,delta)

        # print(BTCVals[0])
        
        optimalBandwidth = estimateOptimalBandwidth(data, T, delta,n)
        print("Optimal Bandwidth,trial:", optimalBandwidth,optimalBandwidth/delta)
        optimalBandwidths.append(optimalBandwidth)
    print(optimalBandwidths)
    print('optimal', np.mean(optimalBandwidths))
    with open('optimal.csv', 'w') as file:
        for h in optimalBandwidths:
            file.write(f"{h}\n")


def MCSims(iterations, loopLength, intervals, seed, bandwidth):

    plotGraphs = True

    loops = int(iterations/loopLength)
    print('loops =',loops)

    quasiDimension = 2
    x = qmc.Halton(quasiDimension, scramble=True, seed=seed)
    data = norm.ppf(x.random(np.array(iterations), workers=-1))
    data[:,1] = rho*data[:,0] + np.sqrt(1-rho**2)*data[:,1]
    # print(data)
    # plt.hist(data[:,0], bins = 1000)
    # plt.show()
    # plt.hist(data[:,1], bins = 1000)
    # plt.show()


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

# MCSims(1000, 1000, 960, 12345, bandwidth)


