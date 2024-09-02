# Monte Carlo Simulations

from math import exp, sqrt, log, tanh, pi
import scipy
from scipy.special import ndtr as N
from scipy.integrate import quad
from scipy.stats import qmc, norm, kurtosis, skew, iqr
from timeit import timeit
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures as cf
import pandas as pd




# Initial Parameters



jumpMean = 0
jumpStd = 1
jumpSize = 0.01
sig = 0.0225
newLim = 0.04**2
jumpLim = 9990.003


etaValues = []
for i in range(45,66):
    etaValues.append(i/10)

w = 30
r = 5

detectionsTotal = 0

stack = deque(maxlen=w)

bandwidth = 0.00707

# inputFile = 'TrainingData.csv'
# outputFile1 = 'profits TrainingData.csv'
# outputFile2 = 'returns TrainingData.csv'

inputFile = 'ValidationData.csv'
outputFile1 = 'profits ValidationData.csv'
outputFile2 = 'returns ValidationData.csv'

# inputFile = 'TestingData.csv'
# outputFile1 = 'profits TestingData.csv'
# outputFile2 = 'returns TestingData.csv'




###################################
#------------------------------------------------------------------------------------------#
###################################
test49 = []
test491 = []
test26 = []



# Batch QMC for multistep with Multithread on Batch 

def dataTest(n):

    plotGraphs = True

    startTime = 480


    dt = 1/(24*20*365)

    
    # Stopping Rule
    stopRule = []

    with open(inputFile, 'r') as file:
        priceData = np.array([float(line.strip()) for line in file])

    newData = []
    for i in range(0, len(priceData) - 1, 3):
        newData.append(priceData[i])



    BTCVals = np.log(newData)
    # plt.plot(BTCVals)
    # plt.show()
    

    print(len(BTCVals))
    DeltaBTC = np.diff(BTCVals)

    print(np.median(abs(DeltaBTC)))
    # plt.hist(abs(DeltaBTC[abs(DeltaBTC)>0.002]), bins=1000)
    # plt.show()

    print('average val =', np.mean(np.exp(BTCVals)))

    # plt.plot(BTCVals[110800:110900])
    # plt.show()

    print('non-zero', np.count_nonzero(DeltaBTC))
    # plt.plot(DeltaBTC)
    # plt.show()

    K = 480
    n = 175200
    a = 0.1
    c = np.sqrt(2/np.pi)
    b = c*np.sqrt(2*np.log(n))
    an = np.sqrt(2*log(n))/c - (log(pi)+log(log(n)))/(2*b)
    print(an,b,c)


    for t in range(K,len(BTCVals)):
        mu = 1/(K-1) * np.sum(DeltaBTC[t-K+1:t-1])
        if i==0 and t == K:
            print('mu =', mu)

        tempVolEst = np.sqrt(1/(K-2) * np.sum(abs(DeltaBTC[t-K:t-2]) * abs(DeltaBTC[t-K+1:t-1])))
        if i==0 and t == K:
            print('tempVolEst =', tempVolEst)

        T = (DeltaBTC[t-1]-mu)/tempVolEst
        if i==0 and t == K:
            print('T =', T)

        beta = -log(-log(1-a))
        if i==0 and t == K:
            print('beta =', beta)

        if b*(np.abs(T)-an) - beta > 0:
            # print('crit vals =', b*(np.abs(T)-an), beta, b*(np.abs(T)-an) - beta, i, t)
            DeltaBTC[t-1] = 0



    fig, ax1 = plt.subplots(figsize=(8, 4))
    plt.plot(DeltaBTC, label='log returns')
    plt.legend()
    ax1.set_xlabel('Time Increments, i')
    ax1.set_ylabel('Log Returns')
    plt.show()
    print('non-zero', np.count_nonzero(DeltaBTC))
    print(len(DeltaBTC))



    newplot2 = False

    if newplot2 == True:

        #  Create a histogram of the log returns
        fig, ax1 = plt.subplots()

        # Plot histogram with density=True
        counts, bins, patches = ax1.hist(DeltaBTC[abs(DeltaBTC)>0], bins=400, density=True, alpha=0.6, color='maroon', edgecolor='maroon')

        # Create a secondary y-axis
        ax2 = ax1.twinx()

        # Generate data for a normal distribution curve
        mu, std = norm.fit(DeltaBTC[abs(DeltaBTC)>0])
        x = np.linspace(min(bins), max(bins), 1000)
        p = norm.pdf(x, mu, std)

        # Further scale down the normal curve
        scale_factor = 0.35  # Adjust this factor until the curve fits better
        p_scaled = p * scale_factor

        # Plot the scaled normal distribution on the secondary y-axis
        ax2.plot(x, p_scaled, 'b', linewidth=0.1)

        # Highlight the area under the normal distribution curve
        ax2.fill_between(x, 0, p_scaled, color='b', alpha=0.2)

        # Manually set the secondary y-axis limits to match the primary y-axis limits
        ax2.set_ylim(bottom=0)

        # Optionally, you can also set the primary y-axis limit to ensure both start at zero
        ax1.set_ylim(bottom=0)

        # Add labels and title
        ax1.set_xlabel('Log Returns')
        ax1.set_ylabel('Frequency')
        ax2.set_ylabel('Scaled Probability Density')
        plt.savefig('returnshist.png', dpi=300, bbox_inches='tight')
        # Show the plot
        plt.show()






    intervals = len(BTCVals)-2

    detection = np.zeros((intervals+1,len(etaValues)))
    detectionTimes = []
    for i in range(len(etaValues)):
        detectionTimes.append([])
    detectionVals = []
    for i in range(len(etaValues)):
        detectionVals.append([])
    


    sig2 = []
    errors = []


    error = []


    sigW = startTime + K

    data = np.square(DeltaBTC)


    weights = np.exp(np.arange(0,startTime,dtype=float)*dt/bandwidth)/bandwidth
    # print('weight =', weights)
    weights = weights / np.sum(weights)

    for t in range(int(sigW+ 1), intervals+1):
        

        # print(np.abs(np.sum(weights*data[t-sigW:t])-np.average(data[t-sigW:t])))
        varEst = np.sum(weights*data[t-startTime-1:t-1])
        stack.append(np.sqrt(varEst/dt))





        sig2.append(varEst/dt)


        WEst = 0
        maxStopRule = 0

        if t >= int(sigW + 1+w):
            for l in range(1, w+1):

                dY = DeltaBTC[t-l]
                WEst += dY/(stack[-l])
            
                if l >= r:
                    newStopRule = abs(WEst)/np.sqrt(l*dt)
                    if newStopRule > maxStopRule:
                        maxStopRule = newStopRule
                        for idx, eta in enumerate(etaValues):

                            if newStopRule > eta:
                                # print('ACTIVATE! t=', t, 'stoprule =', maxStopRule)
                                if detection[t,idx] == 0.0:
                                    detectionVals[idx].append(WEst/np.sqrt(l*dt))
                                    detection[t,idx] = 1
                                    detectionTimes[idx].append(t)

            stopRule.append(maxStopRule)
        if t >= int(sigW + 1+w):
            test49.append(WEst/np.sqrt(l*dt))
        test491.append(maxStopRule)


    errors.append(np.mean(error))
    # if i>4:
    #     print(test49[2000])


    # print(np.mean(test26), np.mean(np.abs(test26)), np.min(np.abs(test26)[np.abs(test26) >np.percentile(np.abs(test26),99)]), 'dbtc')

    # print('eta =', etaValues, 'Detections =', (detection))
    for i in range(len(etaValues)):
            print('eta =', etaValues[i] , 'Detections =', np.sum(detection[:,i]))
    
    global detectionsTotal
    # detectionsTotal += np.sum(detection)


    # Update Matrices
    out = test49
    # sigs = data[0]
    # for i in range(len(sigs)):
    #     plt.plot(sigs[i])
    # plt.show()
    if plotGraphs == True:
        plt.plot(test49, label = 'W Estimate')
        plt.legend()
        plt.show()
        plt.plot(test491, label = 'Stop Rule Value')
        plt.legend()
        plt.show()
        plt.plot(sig2, label = 'variance')
        plt.show()



    print(detectionTimes[0], 'detection times')
    # plt.plot(detectionTimes[-1])
    # plt.show()


    DeltaBTC = np.diff(BTCVals)

    hold = 0
    profit = 0
    holdDirection = 0
    holdPeriod = 1
    etaVal = -1
    returns = []
    positive = []
    portfolios = []
    fee = 0.000
    tradereturns = []
    tradelengths =[]
    portVals = []


    for etaVal in range(len(etaValues)):
        for holdPeriod in range(4,7):
            hold = 0
            profit = 0
            holdDirection = 0
            transCosts = 0
            transNo = 0
            portfolio = 1000

            for time in range(len(detectionTimes[etaVal])):
                if time == 0:
                    hold = 1
                    transNo += 1
                    entry_time = detectionTimes[etaVal][time]
                    if detectionVals[etaVal][time] > 0:
                        holdDirection = 1
                    else:
                        holdDirection = -1

                if hold==0:
                    if detectionTimes[etaVal][time] + holdPeriod >len(BTCVals):
                        break
                    hold = 1
                    transNo += 1
                    entry_time = detectionTimes[etaVal][time]
                    if detectionVals[etaVal][time] > 0:
                        holdDirection = 1
                    else:
                        holdDirection = -1
                
                if hold == 1:
                    if time == len(detectionTimes[etaVal])-1:
                        if detectionTimes[etaVal][time]+holdPeriod >= len(BTCVals):
                            hold = 0
                            profit += (np.exp(BTCVals[-1]) - np.exp(BTCVals[entry_time])) * holdDirection -0.001*(np.exp(BTCVals[-1]) + np.exp(BTCVals[entry_time]))
                            tradelengths.append(exitTime-entry_time)
                            portVals.append(portfolio)
                            portold = portfolio
                            portfolio = (portfolio*(1-fee)/(np.exp(BTCVals[entry_time])))*np.exp(BTCVals[-1])*(1-fee)
                            tradereturns.append(portfolio/portold)
                            transCosts += -0.001*(np.exp(BTCVals[exitTime]) + np.exp(BTCVals[entry_time]))
                            entry_time = 0
                            holdDirection = 0
                            break
                        exitTime = detectionTimes[etaVal][time]+holdPeriod
                        hold = 0
                        profit += (np.exp(BTCVals[exitTime]) - np.exp(BTCVals[entry_time])) * holdDirection -0.001*(np.exp(BTCVals[exitTime]) + np.exp(BTCVals[entry_time]))
                        portold = portfolio
                        tradelengths.append(exitTime-entry_time)
                        portVals.append(portfolio)
                        portfolio = (portfolio*(1-fee)/(np.exp(BTCVals[entry_time])))*np.exp(BTCVals[exitTime])*(1-fee)
                        tradereturns.append(portfolio/portold)
                        transCosts += -0.001*(np.exp(BTCVals[exitTime]) + np.exp(BTCVals[entry_time]))
                        entry_time = 0
                        holdDirection = 0
                    elif detectionTimes[etaVal][time+1] > detectionTimes[etaVal][time]+holdPeriod:
                        exitTime = detectionTimes[etaVal][time]+holdPeriod
                        hold = 0
                        profit += (np.exp(BTCVals[exitTime]) - np.exp(BTCVals[entry_time])) * holdDirection -0.001*(np.exp(BTCVals[exitTime]) + np.exp(BTCVals[entry_time]))
                        portold = portfolio
                        tradelengths.append(exitTime-entry_time)
                        portVals.append(portfolio)
                        # print(exitTime,entry_time,3)
                        portfolio = (portfolio*(1-fee)/(np.exp(BTCVals[entry_time])))*np.exp(BTCVals[exitTime])*(1-fee)
                        tradereturns.append(portfolio/portold)
                        transCosts += -0.001*(np.exp(BTCVals[exitTime]) + np.exp(BTCVals[entry_time]))
                        entry_time = 0
                        holdDirection = 0
                    elif detectionTimes[etaVal][time]+holdPeriod >= len(BTCVals):
                        hold = 0
                        profit += (np.exp(BTCVals[-1]) - np.exp(BTCVals[entry_time])) * holdDirection -0.001*(np.exp(BTCVals[exitTime]) + np.exp(BTCVals[entry_time]))
                        portold = portfolio
                        tradelengths.append(exitTime-entry_time)
                        portVals.append(portfolio)
                        portfolio = (portfolio*(1-fee)/(np.exp(BTCVals[entry_time])))*np.exp(BTCVals[exitTime])*(1-fee)
                        tradereturns.append(portfolio/portold)
                        transCosts += -0.001*(np.exp(BTCVals[exitTime]) + np.exp(BTCVals[entry_time]))
                        entry_time = 0
                        holdDirection = 0
                
                

            returns.append([profit,portfolio,etaValues[etaVal],holdPeriod,transNo,transCosts,profit-transCosts])
            portfolios.append(portfolio)
            if profit > 0:
                positive.append([profit,portfolio,etaValues[etaVal],holdPeriod,transNo,transCosts,profit-transCosts])
            # print('Profit =', profit, 'eta =', etaValues[etaVal], 'hold Period =', holdPeriod, 'transaction costs =', transCosts)

    print('max return =', max(returns))
    print('max & average portfolio =', max(portfolios), np.mean(portfolios))
    plt.plot(portfolios, label='portfolios')
    plt.axhline(y=1000, color='r', linestyle='--', label='0% Return')
    plt.axhline(y=1825, color='black', linestyle='--', label='SPX Return')
    plt.legend()
    plt.show()

    with open(outputFile1, 'w') as file:
        for i in positive:
            file.write(f"{i}\n")
    with open(outputFile2, 'w') as file:
        for i in returns:
            file.write(f"{i}\n")

    averaging = 0
    for i in returns:
        averaging+= i[-1]
    print('average payoff before costs', averaging/len(returns))
    averaging = 0
    for i in returns:
        averaging+= i[0]
    print('average payoff after costs', averaging/len(returns))


    #//////////////////////////////////////////////////////////////#    

    newplot=False
    
    if newplot == True:
        with open('dates.csv', 'r') as file:
            dates = np.array([float(line.strip()) for line in file])

        newDates = []
        for i in range(0, len(dates) - 1, 3):
            newDates.append(dates[i])

        newDates = pd.to_datetime(newDates, unit='s')
        

        from matplotlib.dates import DateFormatter

        fig, ax1 = plt.subplots(figsize=(8, 5))
        # Plotting stock prices
        ax1.plot(newDates[426900:428350], BTCVals[426900:428350], color='maroon', label='BTC/EUR Log Prices')
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('BTC/EUR Log Price', color='maroon', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='maroon')

        # Plotting the statistic with a shaded area
        ax2 = ax1.twinx()
        ax2.plot(newDates[426900:428350], stopRule[426900-(sigW + 1+w):428350-(sigW + 1+w)], 
                'b-', label='Stop Value', alpha=0.6)
        ax2.fill_between(newDates[426900:428350], 0, stopRule[426900-(sigW + 1+w):428350-(sigW + 1+w)], 
                        color='blue', alpha=0.1)
        ax2.set_ylabel('Stop Value', color='b')
        ax2.tick_params(axis='y', labelcolor='b')

        # Formatting the x-axis to display dates/times
        ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
        fig.autofmt_xdate()  # Rotate the x-axis labels for better readability

        plt.axhline(y=4.5, color='black', linestyle='--', label='Stop Value = 4.5')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        # Add a grid
        plt.grid(True, linestyle='--', alpha=0.7)

        # Adding legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.savefig('realData.png', dpi=300, bbox_inches='tight')

        plt.show()

    #//////////////////////////////////////////////////////////////#    




    
    print(len(detectionTimes[0]))
    print(len(detectionVals[0]))

    print('average trade return =', (np.mean(tradereturns)))
    print('average trade length =', np.mean(tradelengths))
    # print(tradelengths)
    # print(tradereturns)
    print(np.product(tradereturns))
    # print(portVals)

    plt.plot(portVals)
    plt.show()



    return out, stopRule



data, stopRule = dataTest(50)
# data, stopRule, error = MCSims(1000, 1000, 6000, 12345, 10)

plt.plot(stopRule)
plt.show()
stopRule = np.array(stopRule)

print(len(stopRule), 'unadjusted length')
stopRule = stopRule[stopRule<10]
print(len(stopRule), 'adjusted length')

plt.hist(stopRule, bins = 1000, label= 'Stoprule Hist')
plt.show()

percentile_99 = np.percentile(stopRule, 99)
plt.hist(stopRule[stopRule > percentile_99], bins = 1000, label= '1pct hist')
plt.show()

 