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
jumpSize = 0.005
sig = 0.0225
newLim = 0.04**2
jumpLim = 9999
startTime = 480

Taun = 1012.0
Tau = 1042.0

pnTheta = 0.25
gradJumpSize = 0.03

dbTheta = 0.75
c=3



pn = True
db = False

etaValues = []
for i in range(34,51):
    etaValues.append(i/10)

w = 30
r = 5

detectionsTotal = 0

stack = deque(maxlen=w)

# bandwidth = 0.00003234
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

    c = 3
    
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

    jumpsDetected = np.count_nonzero(N)
    print('jump =', jumpsDetected)

    if pn == True:

        # Generate price paths
        for t in range(1,intervals+1):
            # print('t=',t)
            for i in range(batchSize):
                sigVals[i,t] = max(sigVals[i,t-1] + k*(gamma-sigVals[i,t-1])*dt + v*np.sqrt(sigVals[i,t-1])*data[i+loop*batchSize,0]*np.sqrt(dt), 0)
                DeltaBTC[i,t-1] = drift*dt + np.sqrt(sigVals[i,t])*data[i+loop*batchSize,1]*np.sqrt(dt) + N[i, t-1] * J[i, t-1] * jumpSize
                if t == Taun:
                    DeltaBTC[i,t-1] = gradJumpSize
                    if i==25:
                        print(DeltaBTC[26,int(Taun-1)], 'Jump')
                if i==25:
                    test261.append(abs(data[i+loop*batchSize,1]))
                    test26.append(DeltaBTC[i,t-1])
                if Taun <= t < Tau:
                    HVals[i,t] = -gradJumpSize * (1-((t-Taun)/(Tau-Taun))**pnTheta)

                DeltaBTC[i,t-1] += HVals[i,t]-HVals[i,t-1]
                # if abs(DeltaBTC[i,t-1])>jumpLim:
                #     DeltaBTC[i,t-1] = 0
                #     # print('jump', t)
                BTCVals[i,t] = BTCVals[i,t-1] + DeltaBTC[i,t-1]

                

            np.random.seed(t)
            np.random.shuffle(data)


    if db == True:
        H= np.full(batchSize,0.0)
        # Generate price paths
        for t in range(1,intervals+1):
            # print('t=',t)
            for i in range(batchSize):
                sigVals[i,t] = max(sigVals[i,t-1] + k*(gamma-sigVals[i,t-1])*dt + v*np.sqrt(sigVals[i,t-1])*data[i+loop*batchSize,0]*np.sqrt(dt), 0)
                DeltaBTC[i,t-1] = drift*dt + np.sqrt(sigVals[i,t])*data[i+loop*batchSize,1]*np.sqrt(dt) + N[i, t-1] * J[i, t-1] * jumpSize
                if i==25:
                    test26.append(DeltaBTC[i,t-1])
                if Taun < t < Tau:
                    HVals[i,t] = (c*((t-Taun)*(1/(24*20)))**(-dbTheta))*dt 
                    H[i] += HVals[i,t]
                    # print(H[26])
                # if t == Tau:
                    # DeltaBTC[i,t-1]-=H[i]

                DeltaBTC[i,t-1] += HVals[i,t]
                # if abs(DeltaBTC[i,t-1])>jumpLim:
                #     DeltaBTC[i,t-1] = 0
                BTCVals[i,t] = BTCVals[i,t-1] + DeltaBTC[i,t-1]

                

            np.random.seed(t)
            np.random.shuffle(data)

    if db == False and pn == False:
        
        # Generate price paths
        for t in range(1,intervals+1):
            # print('t=',t)
            for i in range(batchSize):
                sigVals[i,t] = max(sigVals[i,t-1] + k*(gamma-sigVals[i,t-1])*dt + v*np.sqrt(sigVals[i,t-1])*data[i+loop*batchSize,0]*np.sqrt(dt), 0)
                DeltaBTC[i,t-1] = drift*dt + np.sqrt(sigVals[i,t])*data[i+loop*batchSize,1]*np.sqrt(dt) + N[i, t-1] * J[i, t-1] * jumpSize
                if i==25:
                    test261.append(abs(data[i+loop*batchSize,1]))
                    test26.append(DeltaBTC[i,t-1])
                # if abs(DeltaBTC[i,t-1])>jumpLim:
                #     DeltaBTC[i,t-1] = 0
                BTCVals[i,t] = BTCVals[i,t-1] + DeltaBTC[i,t-1]

                

            np.random.seed(t)
            np.random.shuffle(data)



    #jump removal

    K = 480
    n = 175200
    a = 0.1
    c = np.sqrt(2/np.pi)
    b = c*np.sqrt(2*np.log(n))
    an = np.sqrt(2*log(n))/c - (log(pi)+log(log(n)))/(2*b)
    print(a,b,c)


    for i in range(batchSize):
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

            if b*(np.abs(T)-an) - beta > 0:
                # print('crit vals =', b*(np.abs(T)-an), beta, b*(np.abs(T)-an) - beta, i, t)
                DeltaBTC[i,t-1] = 0


    jumpsUndetected = np.count_nonzero(DeltaBTC[:,:-1]*N)
    print('undetected jumps =', jumpsUndetected,(jumpsDetected-jumpsUndetected)/jumpsDetected)

    # Stopping Rule
    stopRule = []


    sig1 = []
    sig2 = []
    errors = []
    errors2 = []
    sigW = startTime+K

    # print(np.exp(np.arange(int(startTime + 1), intervals+1,dtype=float)*dt/bandwidth))

    

    weights = (1/2)*np.exp(np.arange(0,K,dtype=float)*dt/bandwidth)/bandwidth
    weights = weights / np.sum(weights)

    

    for i in range(batchSize):
        
        error = []
        error2 = []
        if i%500==0:
            print(i)
        
        data = np.square(DeltaBTC[i])


        for t in range(int(sigW + 1), intervals+1):
            #Kernel Method
            varEst = np.sum(weights*data[t-K-1:t-1])
            stack.append(np.sqrt(varEst/dt))

            if i == 25:
                sig1.append(sigVals[i,t-1])
                sig2.append(varEst/dt)

            WEst = 0
            maxStopRule = 0

            if t >= int(sigW + 1+w):
                for l in range(1, w+1):

                    dY = DeltaBTC[i,t-l]
                    WEst += dY/(stack[-l])
                
                    if l >= r:
                        newStopRule = abs(WEst)/np.sqrt(l*dt)
                        if newStopRule > maxStopRule:
                            maxStopRule = newStopRule
                            for idx, eta in enumerate(etaValues):

                                if newStopRule > eta:
                                    if int(Taun)<t<int(Tau) and (pn == True or db == True):

                                        if detection[i,idx] == 0.0:
                                            detectionDelay[i,idx] = t-Taun
                                            detection[i,idx] = 1
                                    else:

                                        badCount[idx] += 1
                                        if falseDetection[i,idx] == 0.0:
                                            ARL[i,idx] = t-startTime
                                            falseDetection[i,idx] = 1
                if maxStopRule>0:
                    stopRule.append(maxStopRule)
            if i == 25:
                test49.append(WEst/np.sqrt(l*dt))
                test491.append(maxStopRule)



            sigEst1 = sigVals[i,t-1]
            error.append(abs(sigEst1-varEst/dt)/sigEst1)
            error2.append((sigEst1-varEst/dt)**2)
            

        errors.append(np.mean(error))
        errors2.append(np.mean(error2))


    test262 = []
    print(np.mean(test26), np.mean(np.abs(test26)), np.min(np.abs(test26)[np.abs(test26) >np.percentile(np.abs(test26),99)]), 'dbtc')
    print('badCount =', badCount)
    if (db == True or pn == True) and np.count_nonzero(detectionDelay)>0:
        for i in range(len(etaValues)):
            print('eta =', etaValues[i] , 'True detections =', np.sum(detection[:,i]))
    for i in range(len(etaValues)):
        print('eta =', etaValues[i] , 'False detections =', np.sum(falseDetection[:,i]))
    if (db == True or pn == True) and np.count_nonzero(detectionDelay)>0:
        for i in range(len(etaValues)):
            print('eta =', etaValues[i] , 'Detection Delay =', np.mean(detectionDelay[detectionDelay[:,i] != 0]))
    if db == False and pn == False and np.count_nonzero(ARL)>0:
        for i in range(len(etaValues)):
            test262.append(ARL[np.nonzero(ARL[:,i]),i])
            print('eta =', etaValues[i] , 'ARL =', np.mean(ARL[np.nonzero(ARL[:,i]),i]))
    
    global detectionsTotal
    detectionsTotal += np.sum(detection) + np.sum(falseDetection)

    print('MSE =', np.mean(errors2))


    return test49, stopRule, sig1, sig2, errors, test491, BTCVals[24], HVals[25], sigVals[25], test261, test262, np.mean(errors2)


# Batch QMC for multistep with Multithread on Batch 

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
    # Update Matrices
    for batch in cf.as_completed(batches):
        data = batch.result()
        out.append(data[0])
        stopRule.append(data[1])

        if plotGraphs == True:
            plt.plot(data[7])
            plt.show()
            plt.plot(data[6])
            plt.show()
            plt.plot(data[0], label = 'W Estimate')
            plt.plot(data[9][startTime+1:], label = 'W Actual')
            plt.legend()
            plt.show()
            plt.plot(data[9], label = 'W Actual')
            plt.legend()
            plt.show()
            plt.plot(data[5], label = 'Stop Rule Value')
            plt.legend()
            plt.show()

            plt.plot(data[8])
            plt.show()
            plt.plot(data[2], label = 'Actual Var')
            plt.plot(data[3], label = 'Estimated Var')
            plt.legend()
            plt.show()

            for i in range(len(data[10])):
                # print(data[10][i][0])
                plt.hist(data[10][i][0], bins=20, label=etaValues[i])
                plt.show()
        print('average error =', np.mean(data[4]))
        print(len(data[2]),len(data[3]))

    out = np.concatenate(out)
    stopRule = np.concatenate(stopRule)
    print(detectionsTotal, 'total detections')

    return out, stopRule, np.mean(data[4]), data[11]


data, stopRule, error, error2 = MCSims(1000, 1000, 1120, 12345, bandwidth)



print(len(stopRule), 'unadjusted length')
stopRule = stopRule[stopRule<10]
print(len(stopRule), 'adjusted length')

plt.hist(stopRule, bins = 1000, label= 'Stoprule Hist')
plt.show()

percentile_99 = np.percentile(stopRule, 99)
plt.hist(stopRule[stopRule > percentile_99], bins = 1000, label= '1pct hist')
plt.show()

