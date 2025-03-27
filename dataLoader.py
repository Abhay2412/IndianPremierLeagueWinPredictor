import pandas

def loadMatchesData():
    return pandas.read_csv("./dataset/matches.csv")

def loadDeliveriesData():
    return pandas.read_csv("./dataset/deliveries.csv")

if __name__ == '__main__':
    dfMatches = loadMatchesData()
    dfDeliveries = loadDeliveriesData()
    print("First 5 records from matches.csv:")
    print(dfMatches.head())
    print("\nFirst 5 records from deliveries.csv:")
    print(dfDeliveries.head())