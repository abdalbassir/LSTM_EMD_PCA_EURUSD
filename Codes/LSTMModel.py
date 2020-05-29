import pandas as pd
import numpy as np
import Settings
import os
from Features import CreateFeatures
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def PCAReduction(df):
    """
    This function applies the PCA on the features
    """
    df = df.dropna()
    (nSamples, nFeatures) = df.shape
    arrayFeatures = df.values
    pca = PCA(n_components=nFeatures)
    pcaFeatures = pca.fit_transform(arrayFeatures)
    explainedRatio = pca.explained_variance_ratio_
    print('Explained Ratio : {}'.format(explainedRatio))
    nPCA = 1+np.sum(np.cumsum(explainedRatio) < Settings.variabilityRatio)
    dfPCAFeatures = pd.DataFrame(
        index=df.index, 
        columns=['PCA_' + str(idx+1) for idx in range(nPCA)], 
        data=pcaFeatures[:, 0:nPCA])
    return dfPCAFeatures

def SplitData():
    """
    This function returns the training data.
    """
    # Get the features
    dfFeatures = pd.read_pickle(Settings.featuresFilename)
    # Applying Z-score normalization
    nDaysScale = Settings.nDaysScale
    dfFeatures = (dfFeatures - dfFeatures.rolling(window=nDaysScale).mean()) / dfFeatures.rolling(window=nDaysScale).std()
    # Add the response (close prices) to the dataframe
    # PCA
    if Settings.applyPCA:
        dfPCAFeatures = PCAReduction(df=dfFeatures)
    else:
        dfPCAFeatures = dfFeatures.copy()
    #
    prices = pd.read_pickle('./Cashe/prices.pickle')
    prices = prices.loc[dfPCAFeatures.index]
    lookForward = Settings.lookForward
    dfPCAFeatures['PreviousPrice'] = prices.iloc[:, 0]
    dfPCAFeatures['Response'] = prices.iloc[:, 0].shift(-lookForward)
    # Drop NaN values
    dfPCAFeatures.dropna(inplace=True)
    dfPCAFeatures.to_pickle(Settings.featuresWithYFilename)
    # Split the data
    inSampleDate = Settings.inSampleDate
    dfTrainSet = dfPCAFeatures[:inSampleDate]
    return dfTrainSet

def GetXY(df, timeSteps):
    """
    Get the independent variables X and the response Y
    """
    Y = df['Response'].values[timeSteps-1:]
    dfInd = df.drop(columns=['Response'], axis=1)
    vals = dfInd.values
    X = np.array([vals[idx-timeSteps:idx] for idx in range(timeSteps, 1+len(vals))])
    return X, Y

def TrainLSTMModel():
    """
    This function creates and trains LSTM model
    """
    # Get Train Dataset
    modelFilename = Settings.modelFilename
    if os.path.exists(modelFilename):
        model = load_model(modelFilename)
        return model
    dfTrainSet = SplitData()
    print(dfTrainSet)
    trainX, trainY = GetXY(df=dfTrainSet, timeSteps=Settings.timeSteps)
    nPts, nSteps, nFeatures = trainX.shape
    print(trainX.shape)
    print(trainY.shape)
    # reshape input to be [samples, time steps, features]
    #trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    #testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # Create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(Settings.nCells, return_sequences=True, input_shape=(Settings.timeSteps, nFeatures)))
    for _ in range(Settings.nLayers-2):
        model.add(LSTM(Settings.nCells, return_sequences=True))
    model.add(LSTM(Settings.nCells, return_sequences=False))
    model.add(Dropout(Settings.dropout))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=Settings.epochs, batch_size=1, verbose=2)
    # Cashe the model
    model.save(Settings.modelFilename)
    return model

def PredictLSTMModel():
    """
    This function forecasts using the LSTM model
    """
    dfFeatures = CreateFeatures()
    lstmFilename = Settings.predictionFilename
    if os.path.exists(lstmFilename):
        dfPrediction = pd.read_pickle(lstmFilename)
        return dfPrediction
    model = TrainLSTMModel()
    dfFeatures = pd.read_pickle(Settings.featuresWithYFilename)
    X, Y = GetXY(df=dfFeatures, timeSteps=Settings.timeSteps)
    # reshape input to be [samples, time steps, features] (already in the good shape)
    # Predict
    YPred = model.predict(X)
    YPred = YPred.reshape(X.shape[0])
    dfPrediction = pd.DataFrame(index=dfFeatures.index[Settings.timeSteps-1:])
    dfPrediction['Y'] = Y
    dfPrediction['YPred'] = YPred
    dfPrediction.to_pickle(lstmFilename)
    return dfPrediction


if __name__ == '__main__':
    # Just for testing this function
    dfPrediction = PredictLSTMModel()
    print('Done')
    
