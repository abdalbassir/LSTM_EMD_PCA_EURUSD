from PyEMD import EMD
import pandas as pd
import Settings
import os
import pickle
from ReadData import ReadData

def ComputeTsEMD(ts):
    '''
    This function computes the levels of EMD and 
    returns as dataframe containing all the EMD levels.
    ts : a time-series 
    '''
    emd = EMD ()
    imfs = emd(ts) 
    N = imfs.shape[0]
    dfImfs = pd.DataFrame(index=ts.index, columns=range(N), data=imfs.T)
    return dfImfs

def GetImfs(globalData):
    filenameImfs = './Cashe/globalData_Imfs.pickle'
    if os.path.exists(filenameImfs):
        with open(filenameImfs, 'rb') as fimfs:
            globalDataImfs = pickle.load(fimfs)
            return globalDataImfs
    globalDataImfs = {}
    for key in globalData:
        df = globalData[key]
        dfImfs = ComputeTsEMD(ts=df.iloc[:, 0])
        globalDataImfs[key] = dfImfs
    # Save the data to the filename as a pickle file
    with open(filenameImfs, 'wb') as fimfs:
        pickle.dump(globalDataImfs, fimfs)
    return globalDataImfs

def DenoiseByEMD():
    globalData = ReadData()
    removeLevel = Settings.denoiseLevel
    filename = './Cashe/globalData_EMD_{}.pickle'.format(removeLevel)
    # if the denoising is not used (removeLevel = 0)
    if removeLevel < 1:
        globalDataEMD = globalData
        with open(filename, 'wb') as f:
            pickle.dump(globalDataEMD, f)
        return globalDataEMD
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            globalDataEMD = pickle.load(f)
        return globalDataEMD
    # Imfs
    globalDataImfs = GetImfs(globalData)
    globalDataEMD = {}
    for key in globalData:
        df = globalData[key]
        dfImfs = globalDataImfs[key]
        NLevels = dfImfs.shape[1]
        tsReons = dfImfs[range(Settings.denoiseLevel, NLevels)].sum(axis=1)
        tsReons.name = df.columns[0]
        globalDataEMD[key] =  pd.DataFrame(tsReons)
    # Save the data to the filename as a pickle file
    with open(filename, 'wb') as f:
        pickle.dump(globalDataEMD, f)
    return globalDataEMD

if __name__ == '__main__':
    globalDataEMD = DenoiseByEMD()
    print('Done')
