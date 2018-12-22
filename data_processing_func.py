# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 21:46:21 2018

@author: david
"""

import numpy as np
import pandas as pd
from pykalman import KalmanFilter
from pyts.image import GADF, GASF, MTF
import pickle


data_len = int(252/4)+1
image_size = 16 # Compress information for every 4 days;
retrain_freq = 5
nFeatures = 4


class CNNSamples:
    
    def __init__(self,
                 method='GADF',
                 data_len=data_len,
                 image_size=image_size,
                 retrain_freq=retrain_freq,
                 nFeatures=nFeatures
                ):
        
        self.data_len = data_len
        self.image_size = image_size
        self.retrain_freq = retrain_freq
        self.samples = []
        self.nSamples = 0
        self.method = method
        self.nFeatures = nFeatures
        
    def getTimeSeriesCNNSample(self, stockData, useSmoothedData=False):
        
        self.permno = stockData.PERMNO.iloc[0]
        data = stockData.drop('PERMNO', axis=1).T
        self.nDays = data.shape[1]
        if self.method=='GADF':
            img = GADF(self.image_size)
        elif self.method=='GASF':
            img = GASF(self.image_size)
        elif self.method=='MTF':
            img = MTF(self.image_size)
        
        for i in range(self.data_len, self.nDays, self.retrain_freq):
            series = data.iloc[:, i-self.data_len:i]
            
            if useSmoothedData:
                Smoother = KalmanFilter(n_dim_state=self.nFeatures, n_dim_obs=self.nFeatures,
                                        em_vars=['transition_matrices','observation_matrices',
                                                 'transition_convariance','observation_convariance',
                                                 'initial_state_mean', 'initial_state_convariance'
                                                ])
                measurements = series.T.values
                Smoother.em(measurements, n_iter=5)
                series, _ = Smoother.smooth(measurements)
                series = series.T
            
            self.samples.append(img.fit_transform(series).T)
        
        self.nSamples = len(self.samples)
        return self


def getTimeSeriesCNNTarget(df, data_len=data_len, retrain_freq=retrain_freq, up_return=0.0125, down_return=-0.0125):
    data = df[['PRC', 'VOL', 'date']]
    nSample = len(data)
    Targets = []
    
    for i in range(data_len, nSample, retrain_freq):
        if data.VOL.iloc[i-retrain_freq:i].values.sum()>0:
            lastVWAP = np.average(data.PRC.iloc[i-retrain_freq:i].values, weights=data.VOL.iloc[i-retrain_freq:i].values)
        else:
            lastVWAP = data.PRC.iloc[i]
        
        if data.VOL.iloc[i:np.min([nSample-1, i+retrain_freq])].values.sum()>0:
            nextVWAP = np.average(data.PRC.iloc[i:np.min([nSample-1, i+retrain_freq])].values,
                                  weights=data.VOL.iloc[i:np.min([nSample-1, i+retrain_freq])].values)
        else:
            nextVWAP = data.PRC.iloc[np.min([nSample-1, i+retrain_freq])]
        
        VWAPReturn = (nextVWAP - lastVWAP)/lastVWAP
        
        _long, _hold, _short = [1, 0, 0], [0, 1, 0], [0, 0, 1]
        
        if VWAPReturn > up_return:
            Targets.append(_long)
        elif VWAPReturn < down_return:
            Targets.append(_short)
        else:
            Targets.append(_hold)
        
    return (df.PERMNO.iloc[0], Targets)



def getTrainingDataFromPath(feature_path=str(image_size)+'-pixel/',
                            target_path='target/',
                            data_type='GADF',
                            image_size=image_size,
                            train_val_size=2/3,
                            train_size=0.75
                            ):
    
    """ This function helps load in all TS-Image samples, and organize them into trainable manner. 
        Parameters explained:
        
        1) feature_path, target_path:
            Path for saved data. 'feature_path': X data; 'target_path': Y data.
            Notice: X data has to be named as 'CNNSamples_1', 'CNNSamples_2', ... etc.
                    Y data has to be named as 'CNNSamples_target'
                    
        2) data_type:
            Takes value only from one of 'GADF' and 'GASF'. Specify the type of TS-Image type.
        
        3) image_size:
            Image_size of each sample.
            
        4) train_val_size:
            Portion of sample taken to do training+validation
        
        5) train_size:
            Portion of sample taken from training+validation data to do training
            
    """
    
    data_file_num = 27
    all_data = []
    for i in range(data_file_num):
        with open(feature_path+'CNNSamples_'+str(i+1), 'rb') as pick:
            all_data += pickle.load(pick)
            pick.close()
    
    X_train, X_val, X_test = [], [], []
    for obj in all_data:
        X_train += (obj.samples[:int((int(obj.nSamples*train_val_size)+1)*train_size)])
        X_val += (obj.samples[int((int(obj.nSamples*train_val_size)+1)*train_size):(int(obj.nSamples*train_val_size)+1)])
        X_test += obj.samples[int(obj.nSamples*train_val_size)+1:]
  
            
    X_train = np.array(X_train).reshape((len(X_train), image_size, image_size, 4))
    X_val = np.array(X_val).reshape((len(X_val), image_size, image_size, 4))
    X_test = np.array(X_test).reshape((len(X_test), image_size, image_size, 4))
    
    with open(target_path+'CNNSamples_target', 'rb') as pick2:
        Ytmp = pickle.load(pick2)
        pick2.close()
    
    Y_train = np.array([y[1][:int((int(obj.nSamples*train_val_size)+1)*train_size)] for y in Ytmp]).reshape((len(X_train), 3))
    Y_val = np.array([y[1][int((int(obj.nSamples*train_val_size)+1)*train_size):(int(obj.nSamples*train_val_size)+1)] for y in Ytmp]).reshape((len(X_val), 3))
    Y_test = np.array([y[1][int(obj.nSamples*train_val_size)+1:] for y in Ytmp]).reshape((len(X_test), 3))
    
    return (X_train, X_val, X_test, Y_train, Y_val, Y_test)




