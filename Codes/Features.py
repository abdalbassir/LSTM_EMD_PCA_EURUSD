import numpy as np
import pandas as pd 
import pickle
import os
from scipy.stats import linregress
import Settings
from ReadData import ReadData
from DenoiseByEMD import DenoiseByEMD


def ExponentialGainOverLoss(prices, periods=14):
    """
    Helper function for RSI and ADX indicators
    """
    returns = (prices - prices.shift(1))
    gains = np.maximum(returns,  0.0)
    losses = np.minimum(returns, 0.0).abs()
    averageLoss = losses.ewm(span = periods).mean()
    averageGain = gains.ewm(span = periods).mean()
    rs = averageGain / averageLoss
    return rs

def RelativeStrengthIndex(prices, periods=14):
    """
    RSI Indicator
    """
    rs = ExponentialGainOverLoss(prices=prices, periods=periods)
    rsi = 100 * rs / (1 + rs)
    rsi[rs == np.inf] = 100.0
    return rsi

def AverageDirectionalIndex(prices, periods=14):
    """
    ADX indicator
    """
    rs = ExponentialGainOverLoss(prices=prices, periods=periods)
    rsi = 100 * rs / (1 + rs)
    rsi[rs == np.inf] = 100.0
    dx = np.abs(2 * rsi - 100)
    dx[rs == np.inf] = 100.0    
    adx = dx.ewm(span = periods).mean()
    return adx

def UlcerIndex(prices, periods=14): 
    """
    Ulcer index indicator
    """
    maxPrices = prices.rolling(window = periods).max()    
    percentDrawdown = 100 * (prices - maxPrices) / maxPrices
    squaredAverage = np.power(percentDrawdown, 2.0).rolling(window = periods).mean()
    ulcerIdx = np.sqrt(squaredAverage)
    return ulcerIdx


def RateOfChange(prices, periods=21):
    """
    Rate of Change indicator
    """
    roc = 100 * prices.pct_change(periods)
    return roc

def MACD(prices, ema1 = 12, ema2 = 26, ema3 = 9): 
    """
    MACD indicator
    """
    closeEMA1 = prices.ewm(span = ema1).mean()
    closeEMA2 = prices.ewm(span = ema2).mean()
    ppo = (closeEMA1 - closeEMA2)
    signalLine = ppo.ewm(span = ema3).mean()
    return signalLine

def BollingerBands(prices, periods=14, std=2): 
    """
    Bollinger Bands indicator
    """
    meanPrices = prices.rolling(window = periods).mean()   
    stdPrices = prices.rolling(window = periods).std()   
    bbl = (prices - meanPrices) / (std * stdPrices)
    return bbl

def CommodityChannelIndex(prices, highPrices, lowPrices, periods=20, constant=0.015):
    """
    CCI Indicator
    """
    typicalPrices = (prices + highPrices + lowPrices) / 3.0
    meanPrices = typicalPrices.rolling(window = periods).mean()   
    stdPrices = typicalPrices.rolling(window = periods).std()
    cci = (typicalPrices - meanPrices) / (constant * stdPrices)
    return cci

def AverageTrueRange(prices, highPrices, lowPrices, periods=10):
    """
    ATR indicator
    """
    tr1 = np.maximum(highPrices - lowPrices, np.abs(highPrices - prices.shift(1)))
    tr = np.maximum(tr1, np.abs(lowPrices - prices.shift(1)))
    atr = tr.ewm(span=periods).mean()
    return atr

def StochasticOscillator(prices, highPrices, lowPrices, periods=10):
    """
    Stochastic Oscillator
    """
    lowest = lowPrices.rolling(window=periods).min()
    highest = highPrices.rolling(window=periods).max()
    percentK = 100 * ((prices - lowest) / (highest - lowest))
    percentD = percentK.rolling(window=3).mean()
    return percentK, percentD


def get_slope(array):
    """
    This function calculates time series slope 
    """
    y = np.array(array)
    x = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    return slope


def ComputeAllFeatures():
    """
    This function computes and cashes all the features
    """
    removeLevel = Settings.denoiseLevel
    emdDir = Settings.emdDir
    filename = './Cashe/globalData_EMD_{}.pickle'.format(removeLevel)
    with open(filename, 'rb') as f:
        globalDataEMD = pickle.load(f)
    dfClosePrice = globalDataEMD['close']
    dfOpenPrice = globalDataEMD['open'] 
    dfHighPrice = globalDataEMD['high'] 
    dfLowPrice = globalDataEMD['low'] 
    # Close
    dfClosePrice.to_pickle('{}/C_EMD_{}.pickle'.format(emdDir, removeLevel))
    # Open
    dfOpenPrice.to_pickle('{}/O_EMD_{}.pickle'.format(emdDir, removeLevel))
    # High
    dfHighPrice.to_pickle('{}/H_EMD_{}.pickle'.format(emdDir, removeLevel))
    # Low
    dfLowPrice.to_pickle('{}/L_EMD_{}.pickle'.format(emdDir, removeLevel))
    # Close to Low
    c2l = (dfClosePrice / dfLowPrice)
    c2l.to_pickle('{}/C2L_EMD_{}.pickle'.format(emdDir, removeLevel))
    # Close to Open
    c2o = (dfClosePrice / dfOpenPrice)
    c2o.to_pickle('{}/C2O_EMD_{}.pickle'.format(emdDir, removeLevel))
    # High to Low
    h2l = (dfHighPrice / dfLowPrice)
    h2l.to_pickle('{}/H2L_EMD_{}.pickle'.format(emdDir, removeLevel))
    # High to Open
    h2o = (dfHighPrice / dfOpenPrice)
    h2o.to_pickle('{}/H2O_EMD_{}.pickle'.format(emdDir, removeLevel))
    # High to Close
    h2c = (dfHighPrice / dfClosePrice)
    h2c.to_pickle('{}/H2C_EMD_{}.pickle'.format(emdDir, removeLevel))
    # RSI
    rsiFilename = '{}/RSI_EMD_{}.pickle'.format(emdDir, removeLevel)
    if not os.path.exists(rsiFilename):
        rsi = RelativeStrengthIndex(prices=dfClosePrice, periods=14)
        rsi.to_pickle(rsiFilename)
    # BBL
    bblFilename = '{}/BBL_EMD_{}.pickle'.format(emdDir, removeLevel)
    if not os.path.exists(bblFilename):
        bbl= BollingerBands(prices=dfClosePrice, periods=5, std=2)
        bbl.to_pickle(bblFilename)
    # ADX
    adxFilename = '{}/ADX_EMD_{}.pickle'.format(emdDir, removeLevel)
    if not os.path.exists(adxFilename):
        adx = AverageDirectionalIndex(prices=dfClosePrice, periods=5)
        adx.to_pickle(adxFilename)
    # ATR
    atrFilename = '{}/ATR_EMD_{}.pickle'.format(emdDir, removeLevel)
    if not os.path.exists(atrFilename):
        atr =  AverageTrueRange(
            prices=dfClosePrice, 
            highPrices=dfHighPrice, 
            lowPrices=dfLowPrice, 
            periods=7
        )
        atr.to_pickle(atrFilename)
    # CCI
    cciFilename = '{}/CCI_EMD_{}.pickle'.format(emdDir, removeLevel)
    if not os.path.exists(cciFilename):
        cci =  CommodityChannelIndex(
            prices=dfClosePrice, 
            highPrices=dfHighPrice, 
            lowPrices=dfLowPrice,
            periods=5
        )
        cci.to_pickle(cciFilename)
    # Log Return
    logPriceFilename = '{}/LogPrice_EMD_{}.pickle'.format(emdDir, removeLevel)
    if not os.path.exists(logPriceFilename):
        logPrice = np.log(dfClosePrice)
        logPrice.to_pickle(logPriceFilename)
    # percentK, percentD
    percentKFilename = '{}/PercentK_EMD_{}.pickle'.format(emdDir, removeLevel)
    if not os.path.exists(percentKFilename):
        percentK, percentD = StochasticOscillator(
            prices=dfClosePrice, 
            highPrices=dfHighPrice, 
            lowPrices=dfLowPrice,
            periods=5
        )
        percentK.to_pickle(percentKFilename)
        percentD.to_pickle('{}/PercentD_EMD_{}.pickle'.format(emdDir, removeLevel))
    # MOM
    momFilename = '{}/MOM_EMD_{}.pickle'.format(emdDir, removeLevel)
    if not os.path.exists(momFilename):
        mom = RateOfChange(prices=dfClosePrice, periods=1)
        mom.to_pickle(momFilename)

def CreateFeatures():
    """
    This function combines all the features in a single dataframe
    """
    globalDataEMD = DenoiseByEMD()
    emdDir = Settings.emdDir
    ComputeAllFeatures()
    removeLevel = Settings.denoiseLevel
    features = Settings.features
    featuresFilename = Settings.featuresFilename
    if os.path.exists(featuresFilename):
        dfFeatures = pd.read_pickle(featuresFilename)
        return dfFeatures
    dfFeatures = pd.DataFrame()
    for key in features:
        keyFilename = '{}/{}_EMD_{}.pickle'.format(emdDir, key, removeLevel)
        dfFeatures['{}_EMD_{}'.format(key, removeLevel)] = pd.read_pickle(keyFilename).iloc[:, 0]
    dfFeatures.to_pickle(featuresFilename)
    return dfFeatures


if __name__ == '__main__':
    # Just for testing this function
    dfFeatures = CreateFeatures()
    print(dfFeatures)
    print('Done')
    