
# EMD Denoising Parameters
# if denoiseLevel = 0, the features are computed on the raw data
# if denoiseLevel = 1, one level of noise (high-frequency component) is removed
#  if denoiseLevel = 2, two levels of noise are removed, and so on ..
denoiseLevel = 1 

# Features for LSTM Model
inSampleDate = '2018-02-01' # From the begining until this date is for training
features = [
    'ADX', 'ATR', 'BBL', 'C2L', 'C2O', 'CCI', 'C',
    'H2C', 'H2L', 'H2O', 'H', 'L', 'LogPrice',
    'MOM', 'O', 'PercentD', 'PercentK', 'RSI']

# PCA Parameteres
applyPCA = True 
variabilityRatio = 0.95 # 95 % of the variability of the original data is conserved

# LSTM Parameters
lookForward = 1 # Forecast 1 day ahead
nDaysScale = 252 # Rolling z-score normalization is used
timeSteps =  10 #5
nCells = 50 
epochs = 35 
nLayers = 5
dropout = 0.3

# Filenames for Cashing, not necessary to change these variables
lstmStr = 'PCA_{}_{}_TS_{}_NCells_{}_Epochs_{}_NLayers_{}_Dropout_{}'.format(
    applyPCA, variabilityRatio, timeSteps, nCells, epochs, nLayers, dropout)
inputFilename = './Input/EURUSD.1Day.csv'
featuresStr = '.'.join(features)
cashDir = './Cashe'
featuresDir = '{}/Features'.format(cashDir)
emdDir = '{}/EMD_{}'.format(featuresDir, denoiseLevel)
modelDir = '{}/Model'.format(cashDir)
resultsDir = '{}/Results'.format(cashDir)
pricesFilename = '{}/prices.pickle'.format(cashDir)
featuresFilename = './Cashe/Features/{}_EMD_{}.pickle'.format(featuresStr, denoiseLevel)
featuresWithYFilename = './Cashe/Features/{}_Response_EMD_{}_PCA_{}_{}.pickle'.format(
    featuresStr, denoiseLevel, applyPCA, variabilityRatio)
modelFilename = './Cashe/Model/Model_{}_EMD_{}_{}.h5'.format(
    featuresStr, denoiseLevel, lstmStr)
equityFilename = './Cashe/Results/EquityCurve_{}_EMD_{}_{}'.format(
    featuresStr, denoiseLevel, lstmStr)
predictionFilename = './Cashe/Results/Prediction_{}_EMD_{}_{}'.format(
    featuresStr, denoiseLevel, lstmStr)
equityFigureFilename = equityFilename + '.png'
equityPickleFilename = equityFilename + '.pickle'
