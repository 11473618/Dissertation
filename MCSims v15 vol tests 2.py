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
jumpLim = 9990.01
startTime = 480

Taun = 1440.0
Tau = 1520.0

Taun2 = 1540.0
Tau2 = 1620.0

pnTheta = 0.45
gradJumpSize = -0.014
# gradJumpSize = 0

gradJumpSize2 = 0.014
# gradJumpSize2 = 0

dbTheta = 0.65
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

# bandwidth = 0.00011
bandwidth = 0.0108
# bandwidth = 0.002848776629694912




###################################
#------------------------------------------------------------------------------------------#
###################################
test49 = []
test491 = []
test26 = []
test261 = []


# Define Batch Process for multistep QMC
def batchProcess(data, loop, batchSize, intervals, bandwidth):


    
    badCount = np.zeros(len(etaValues))
    detection = np.zeros((batchSize, len(etaValues)))
    detection2 = np.zeros((batchSize, len(etaValues)))
    falseDetection = np.zeros((batchSize, len(etaValues)))
    ARL = np.zeros((batchSize, len(etaValues)))
    detectionDelay = np.zeros((batchSize, len(etaValues)))
    detectionDelay2 = np.zeros((batchSize, len(etaValues)))

    dt = 1/(24*20*365)

    BTCVals = np.full((batchSize,intervals+1),BTC0)
    DeltaBTC = np.full((batchSize,intervals+1),0.0)
    sigVals = np.full((batchSize,intervals+1),sig)
    HVals = np.full((batchSize,intervals+1),0.0)


    np.random.seed(12345)
    N = np.random.poisson(jumpActivity * dt, (batchSize, intervals))
    J = np.random.normal(jumpMean, jumpStd, (batchSize, intervals))

    if pn == True:

        # Generate price paths
        for t in range(1,intervals+1):
            if t%100 == 0:
                print('t=',t)
            for i in range(batchSize):
                W = np.random.normal(jumpMean, jumpStd)
                W2 = np.random.normal(jumpMean, jumpStd)
                W3 = rho*W + np.sqrt(1-rho**2)* W2
                sigVals[i,t] = max(sigVals[i,t-1] + k*(gamma-sigVals[i,t-1])*dt + v*np.sqrt(sigVals[i,t-1])*W*np.sqrt(dt), 0)
                DeltaBTC[i,t-1] = drift*dt + np.sqrt(sigVals[i,t])*W3*np.sqrt(dt) + N[i, t-1] * J[i, t-1] * jumpSize
                if t == Taun:
                    DeltaBTC[i,t-1] = gradJumpSize
                if t == Taun2:
                    DeltaBTC[i,t-1] = gradJumpSize2
                    if i==10:
                        print(DeltaBTC[0,int(Taun-1)], 'Jump')
                if i==10:
                    test261.append(abs(W3))
                    test26.append(DeltaBTC[i,t-1])
                if Taun <= t < Tau:
                    # print(i,t,-gradJumpSize2 * (1-((t-Taun)/(Tau2-Taun))**pnTheta))
                    HVals[i,t] = -gradJumpSize * (1-((t-Taun)/(Tau-Taun))**pnTheta)
                if Taun2 <= t < Tau2:
                    # print(i,t,-gradJumpSize2 * (1-((t-Taun2)/(Tau2-Taun2))**pnTheta))
                    HVals[i,t] = -gradJumpSize2 * (1-((t-Taun2)/(Tau2-Taun2))**pnTheta)



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
            if t%100 == 0:
                print('t=',t)
            for i in range(batchSize):
                sigVals[i,t] = max(sigVals[i,t-1] + k*(gamma-sigVals[i,t-1])*dt + v*np.sqrt(sigVals[i,t-1])*J[i, t-1]*np.sqrt(dt), 0)
                DeltaBTC[i,t-1] = drift*dt + np.sqrt(sigVals[i,t])*-J[i, t-1]*np.sqrt(dt) + N[i, t-1] * J[i, t-1] * jumpSize
                if i==25:
                    test26.append(DeltaBTC[i,t-1])
                if Taun < t < Tau:
                    HVals[i,t] = (c*((t-Taun)*(1/(24*60)))**(-dbTheta))*dt 
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
                W = np.random.normal(jumpMean, jumpStd)
                W2 = np.random.normal(jumpMean, jumpStd)
                W3 = rho*W + np.sqrt(1-rho**2)* W2
                sigVals[i,t] = max(sigVals[i,t-1] + k*(gamma-sigVals[i,t-1])*dt + v*np.sqrt(sigVals[i,t-1])*W*np.sqrt(dt), 0)
                DeltaBTC[i,t-1] = drift*dt + np.sqrt(sigVals[i,t])*W3*np.sqrt(dt) + N[i, t-1] * J[i, t-1] * jumpSize
                if i==25:
                    test261.append(abs(data[i+loop*batchSize,1]))
                    test26.append(DeltaBTC[i,t-1])
                # if abs(DeltaBTC[i,t-1])>jumpLim:
                #     DeltaBTC[i,t-1] = 0
                BTCVals[i,t] = BTCVals[i,t-1] + DeltaBTC[i,t-1]

                

            np.random.seed(t)
            np.random.shuffle(data)



    #jump removal

    K = startTime
    n = 175200
    a = 0.1
    c = np.sqrt(2/np.pi)
    b = c*np.sqrt(2*np.log(n))
    an = np.sqrt(2*log(n))/c - (log(pi)+log(log(n)))/(2*b)
    print(a,b,c)


    for i in range(batchSize):
        if i%200 == 0:
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

            if b*(np.abs(T)-an) - beta > 0:
                # print('crit vals =', b*(np.abs(T)-an), beta, b*(np.abs(T)-an) - beta, i, t)
                DeltaBTC[i,t-1] = 0


    # Stopping Rule
    stopRule = []


    sig1 = []
    sig2 = []

    errors = []
    errors2 = []
    sigW = startTime+K

    # print(np.exp(np.arange(int(startTime + 1), intervals+1,dtype=float)*dt/bandwidth))

    

    weights = (1/2)*np.exp(np.arange(0,K,dtype=float)*dt/bandwidth)/bandwidth
    # print('weight =', weights)
    weights = weights / np.sum(weights)

    errors3 = []
    errors4 = []
    errors5 = []
    errors6 = []


    for i in range(batchSize):
        
        error = []
        error2 = []
        if i%200 == 0:
            print('i=',i)
        
        data = np.square(DeltaBTC[i])

        for t in range(int(sigW), intervals+1):
            #Kernel Method
            varEst = np.sum(weights*data[t-K-1:t-1])
            stack.append(np.sqrt(varEst/dt))

            if i == 10:
                sig1.append(np.sqrt(sigVals[i,t-1]))
                sig2.append(np.sqrt(varEst/dt))

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
                                    # if i==10:
                                    #     print(newStopRule,maxStopRule)
                                    if int(Taun)<t<int(Tau) and (pn == True or db == True):
                                        # print('ACTIVATE! t=', t, 'stoprule =', maxStopRule)
                                        if detection[i,idx] == 0.0:
                                            detectionDelay[i,idx] = t-Taun
                                            detection[i,idx] = 1
                                    elif int(Taun2)<t<int(Tau2) and (pn == True or db == True):
                                        # print('ACTIVATE! t=', t, 'stoprule =', maxStopRule)
                                        if detection2[i,idx] == 0.0:
                                            detectionDelay2[i,idx] = t-Taun2
                                            detection2[i,idx] = 1
                                    else:
                                        # if i ==10:
                                        #     print('bad t=', t, 'stoprule =', newStopRule)
                                        badCount[idx] += 1
                                        if falseDetection[i,idx] == 0.0:
                                            ARL[i,idx] = t-startTime
                                            falseDetection[i,idx] = 1
                if maxStopRule>0:
                    stopRule.append(maxStopRule)
                if i == 10:
                    test49.append(WEst/np.sqrt(l*dt))
                    # print(t,maxStopRule)
                    test491.append(maxStopRule)


            #error calc
            sigEst1 = sigVals[i,t-1]
            error.append(abs(sigEst1-varEst/dt)/sigEst1)
            error2.append((np.sqrt(sigEst1)-np.sqrt(varEst/dt))**2)
            if int(Taun)-480<=t<int(Taun):
                errors3.append((np.sqrt(sigEst1)-np.sqrt(varEst/dt))**2)
            elif int(Taun)<=Taun<int(Taun)+480:
                errors4.append((np.sqrt(sigEst1)-np.sqrt(varEst/dt))**2)
            if int(Taun)-480<=t<int(Taun):
                errors5.append(np.sqrt(varEst/dt)-(np.sqrt(sigEst1)))
            elif int(Taun)<=Taun<int(Taun)+480:
                errors6.append(np.sqrt(varEst/dt)-(np.sqrt(sigEst1)))

        errors.append(np.mean(error))
        errors2.append(np.mean(error2))
        # if i>4:
        #     print(test49[2000])

    test262 = []
    print(np.mean(test26), np.mean(np.abs(test26)), np.min(np.abs(test26)[np.abs(test26) >np.percentile(np.abs(test26),99)]), 'dbtc')
    print('badCount =', badCount)
    if (db == True or pn == True) and np.count_nonzero(detectionDelay)>0:
        for i in range(len(etaValues)):
            print('eta =', etaValues[i] , 'True detections =', np.sum(detection[:,i]))
    if (db == True or pn == True) and np.count_nonzero(detectionDelay2)>0:
        for i in range(len(etaValues)):
            print('eta =', etaValues[i] , 'True detections 2 =', np.sum(detection2[:,i]))
    for i in range(len(etaValues)):
        print('eta =', etaValues[i] , 'False detections =', np.sum(falseDetection[:,i]))
    if (db == True or pn == True) and np.count_nonzero(detectionDelay)>0:
        for i in range(len(etaValues)):
            print('eta =', etaValues[i] , 'Detection Delay =', np.mean(detectionDelay[detectionDelay[:,i] != 0]))
    if (db == True or pn == True) and np.count_nonzero(detectionDelay2)>0:
        for i in range(len(etaValues)):
            print('eta =', etaValues[i] , 'Detection Delay 2 =', np.mean(detectionDelay2[detectionDelay2[:,i] != 0]))
    if db == False and pn == False and np.count_nonzero(ARL)>0:
        for i in range(len(etaValues)):
            test262.append(ARL[np.nonzero(ARL[:,i]),i])
            print('eta =', etaValues[i] , 'ARL =', np.mean(ARL[np.nonzero(ARL[:,i]),i]))
    
    global detectionsTotal
    detectionsTotal += np.sum(detection) + np.sum(falseDetection)

    print('MSE =', np.mean(errors2))

    print(np.mean(errors3), np.std(errors3))
    print(np.mean(errors4), np.std(errors4))
    print(np.mean(errors5), np.std(errors5))
    print(np.mean(errors6), np.std(errors6))


    return test49, stopRule, sig1, sig2, errors, test491, BTCVals[11], HVals[10], sigVals[10], test261, test262, np.mean(errors2)


# Batch QMC for multistep with Multithread on Batch 

def MCSims(iterations, loopLength, intervals, seed, bandwidth):

    plotGraphs = False

    sigW = startTime+480 + 1 + w

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
    # Update Matrices
    for batch in cf.as_completed(batches):
        data = batch.result()
        out.append(data[0])
        stopRule.append(data[1])
        # sigs = data[0]
        # for i in range(len(sigs)):
        #     plt.plot(sigs[i])
        # plt.show()
        if plotGraphs == True:
            plt.plot(data[7])
            plt.show()
            plt.figure(figsize=(8, 4))
            plt.plot(data[6][960:])
            plt.axvline(x=Taun-960, color='black', linestyle='--', label='Drift Burst Start')
            plt.axvline(x=Tau-960, color='grey', linestyle='--', label='Drift Burst End')
            plt.axvline(x=Taun2-960, color='darkgreen', linestyle='--', label='2nd Burst Start')
            plt.axvline(x=Tau2-960, color='forestgreen', linestyle='--', label='2nd Burst End')
            plt.xlabel('Time Intervals, i', fontsize=12)
            plt.ylabel('Log Price', fontsize=12)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='best', fontsize=10)
            plt.savefig('logprice1.png', dpi=300, bbox_inches='tight')
            plt.show()
            plt.plot(data[0], label = 'W Estimate')
            plt.plot(data[9][startTime+1:], label = 'W Actual')
            plt.legend()
            plt.show()
            plt.plot(data[9], label = 'W Actual')
            plt.legend()
            plt.show()
            plt.figure(figsize=(8, 4))
            plt.plot(data[5], label = 'Stop Rule Value')
            plt.axvline(x=Taun-sigW, color='black', linestyle='--', label='Drift Burst Start')
            plt.axvline(x=Tau-sigW, color='grey', linestyle='--', label='Drift Burst End')
            plt.axvline(x=Taun2-sigW, color='darkgreen', linestyle='--', label='2nd Burst Start')
            plt.axvline(x=Tau2-sigW, color='forestgreen', linestyle='--', label='2nd Burst End')
            plt.xlabel('Time Intervals, i', fontsize=12)
            plt.ylabel('Stop Rule Value', fontsize=12)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='best', fontsize=10)
            plt.savefig('stoprule1.png', dpi=300, bbox_inches='tight')
            plt.show()
            # plt.plot(data[9], label = 'W Actual')
            # plt.plot(data[5], label = 'Stop Rule Value')
            # plt.legend()
            # plt.show()
            plt.plot(data[8])
            plt.show()
            plt.figure(figsize=(8, 4))
            plt.plot(data[2], label = 'Actual Volatility', linewidth=2)
            plt.plot(data[3], label = 'Estimated Volatility', color='darkred', linewidth=2)
            plt.axvline(x=Taun-sigW, color='black', linestyle='--', label='Drift Burst Start')
            plt.axvline(x=Tau-sigW, color='grey', linestyle='--', label='Drift Burst End')
            plt.axvline(x=Taun2-sigW, color='darkgreen', linestyle='--', label='2nd Burst Start')
            plt.axvline(x=Tau2-sigW, color='forestgreen', linestyle='--', label='2nd Burst End')
            plt.xlabel('Time Intervals, i', fontsize=12)
            plt.ylabel('Volatility', fontsize=12)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='best', fontsize=10)
            plt.savefig('variance2.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # plt.hist(data[4])
            # plt.show()
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


# print(MCSims(10, 10, 200, 12345))



# errors = []
# for n in range(10,201,10):
#     data, stopRule, error = MCSims(100, 100, 12000, 12345,n)
#     errors.append(error)


# print(errors)
# plt.plot(errors)
# plt.show()

#////////////////////////////////////////////////////////////////////////////////#

# errors2 = []
# hvals = []
# for h in range(11,12):
#     # h = bandwidth + h/1000
#     h = h/10000
#     hvals.append(h)
#     h = bandwidth
#     data, stopRule, error, error2 = MCSims(30, 30, 16390, 12345, h)
#     errors2.append(error2)

# print(errors2)
# print(hvals)
# plt.plot(errors2)
# plt.show()

#////////////////////////////////////////////////////////////////////////////////#

# data, stopRule, error, error2 = MCSims(30, 30, 16390, 12345, bandwidth)



data, stopRule, error, error2 = MCSims(5000, 5000, 2400, 12345, bandwidth)
# data, stopRule, error, error2 = MCSims(1000, 1000, 1350, 12345, bandwidth)

print(len(stopRule), 'unadjusted length')
stopRule = stopRule[stopRule<10]
print(len(stopRule), 'adjusted length')

plt.hist(stopRule, bins = 1000, label= 'Stoprule Hist')
plt.show()

percentile_99 = np.percentile(stopRule, 99)
plt.hist(stopRule[stopRule > percentile_99], bins = 1000, label= '1pct hist')
plt.show()

