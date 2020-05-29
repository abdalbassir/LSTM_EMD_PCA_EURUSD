import pandas as pd
import numpy as np
import Settings
import os
from LSTMModel import PredictLSTMModel
import matplotlib.pyplot as plt


def StrategyDevelopment():
    """
    This function creates and cashes the results (equity curve)
    """
    capital = 1e7
    figsize=(12, 8)
    dfPrediction = PredictLSTMModel()
    dfPrices = pd.read_pickle(Settings.pricesFilename)
    dfPrices = dfPrices.loc[dfPrediction.index]
    tsSignals = np.sign(dfPrediction['YPred'] - dfPrices.iloc[:, 0])
    dfSignals = pd.DataFrame(index=dfPrices.index, columns=dfPrices.columns, data=tsSignals)
    dfPositions = capital * dfSignals / dfPrices
    dfPnls = dfPositions.shift(1) * dfPrices.diff()
    tsReturns = 100 * dfPnls.sum(axis=1) / capital
    tsReturns.to_pickle(Settings.equityPickleFilename)
    # Equity Curve Plot
    XInSample = [pd.to_datetime(Settings.inSampleDate), pd.to_datetime(Settings.inSampleDate)]
    YInSample = [0, max(tsReturns.cumsum())]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    tsReturns.cumsum().plot(
        ax=ax, 
        title='Equity Curve : LSTM Model - Denoise ({}) - PCA : {}'.format(
            Settings.denoiseLevel, Settings.applyPCA))
    ax.plot(XInSample, YInSample, '-r', linewidth=3)
    plt.xlabel('Date')
    plt.ylabel('Returns [%]')
    plt.grid()
    plt.savefig(Settings.equityFigureFilename) 
    plt.close()




if __name__ == '__main__':
    # Run the simulation
    StrategyDevelopment()
    print('Done')
    
