import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt
import glob
import os

# Path to the directory containing the CSV files
path = "BTCVals\old\FinalSample"  # Update this with your actual path

# Get all file paths that match the pattern
all_files = sorted(glob.glob(os.path.join(path, "BTCEUR-trades-2024-*.csv")))

# Define the output file name
outputFile = 'TestingData.csv'


# Initialize a list to store the average prices for each second
averagePrices = []

for filePath in all_files:

    print('Processing', filePath)
    # Import Data
    # data = genfromtxt('BTCEUR-trades-2024-06-01.csv', delimiter=',')
    data = genfromtxt(filePath, delimiter=',')
    print(data)

    modifiedData = data[:, :-2]


    # Shorten timestamp to seconds (from milliseconds)
    modifiedData[:, -1] = modifiedData[:, -1] // 1000

    # Set start and end timestamps from data
    startTimestamp = int(modifiedData[0 , -1])
    endTimestamp = int(modifiedData[-1 , -1]-59)

    dataPointer = 0
    dataLength = len(modifiedData)

    # Initialize a variable to store the previous second's price
    previousPrice = None

    for currentTimestamp in range(startTimestamp, endTimestamp,60):
        print(currentTimestamp)
         # Initialize a list to accumulate prices for the current timestamp
        prices = []

        # Move the pointer through the sorted data
        while dataPointer < dataLength and modifiedData[dataPointer, -1] < currentTimestamp + 60:
            # Accumulate prices
            prices.append(modifiedData[dataPointer, 1])
            dataPointer += 1

        if prices:
            # Calculate the average price for the current minute
            averagePrice = np.mean(prices)
        else:
            # Use the previous minute's price if no rows match the current timestamp
            averagePrice = previousPrice

        # Store the result
        averagePrices.append((currentTimestamp, averagePrice))

        # Update the previousPrice to the current averagePrice for the next iteration
        if prices:
            previousPrice = averagePrice


    
# Convert the list to a numpy array if needed
averagePrices = np.array(averagePrices)



# Print the results
print(averagePrices)






# Define the file name you want to write to
# outputFile = 'average prices Apr.csv'

# Open the file in write mode and write the contents of averagePrices to it
with open(outputFile, 'w') as file:
    for price in averagePrices[:,-1]:
        file.write(f"{price}\n")


plt.plot(averagePrices[:,-1])
plt.show()
print(startTimestamp, endTimestamp)