import pandas as pd
import pickle
import os
import Settings 

def Initialize():
    """
    This function checks and creates the necessary repositories
    """
    cashDir = Settings.cashDir
    featuresDir = Settings.featuresDir
    modelDir = Settings.modelDir
    resultsDir = Settings.resultsDir
    emdDir = Settings.emdDir
    if not os.path.isdir(cashDir):
        os.mkdir(cashDir)
    if not os.path.isdir(featuresDir):
        os.mkdir(featuresDir)
    if not os.path.isdir(modelDir):
        os.mkdir(modelDir)
    if not os.path.isdir(resultsDir):
        os.mkdir(resultsDir)
    if not os.path.isdir(emdDir):
        os.mkdir(emdDir)

def ReadData():
    """
    This function reads the original data from the Input and
    create a global dictionary containing the close, open, high and low prices
    as dataframes.
    """
    Initialize()
    filename = './Cashe/globalData.pickle'
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            globalData = pickle.load(f)
        return globalData
    globalData = {}
    df = pd.read_csv(Settings.inputFilename, parse_dates=True, index_col=0)
    df = df.reset_index().drop_duplicates()
    df.set_index('Time', inplace = True)
    df.index = pd.DatetimeIndex(df.index.date)
    securityName = 'EURUSD'
    # Close Prices
    tsClosePrice = df.close
    tsClosePrice.name = securityName
    dfClosePrice = pd.DataFrame(tsClosePrice)
    dfClosePrice.to_pickle(Settings.pricesFilename)
    globalData['close'] = dfClosePrice
    # Open Prices 
    tsOpenPrice = df.open
    tsOpenPrice.name = securityName
    dfOpenPrice = pd.DataFrame(tsOpenPrice)
    globalData['open'] = dfOpenPrice
    # High Prices 
    tsHighPrice = df.high
    tsHighPrice.name = securityName
    dfHighPrice = pd.DataFrame(tsHighPrice)
    globalData['high'] = dfHighPrice
    # Low Prices
    tsLowPrice = df.low
    tsLowPrice.name = securityName
    dfLowPrice = pd.DataFrame(tsLowPrice)
    globalData['low'] = dfLowPrice

    # Save the data to the filename as a pickle file
    with open(filename, 'wb') as f:
        pickle.dump(globalData, f)
    return globalData



if __name__ == '__main__':
    # Just for testing this function
    globalData = ReadData()
    print('Done')
