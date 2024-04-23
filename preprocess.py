import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import filter
from ecgdetectors import Detectors
from scipy.stats import zscore
import wfdb

#set main parameters
frequency = 128
period = float(1 /frequency)
order=5
detectors = Detectors(frequency)

def PlotFigure(Title, DataToPlot, xLabel, yLabel):
    plt.figure(figsize=(30, 10))
    plt.title(Title, fontsize=24)
    plt.plot(DataToPlot,color="#51A6D8", linewidth=1)
    plt.xlabel(xLabel, fontsize=16)
    plt.ylabel(yLabel, fontsize=16)
    plt.show()

def GetRecordData(Record):
    RawData, fields = wfdb.rdsamp(Record)
    RawData = RawData[:,0]

    FilteredData = filter.final_filter(RawData, frequency, order)
    
    # different readers are offered to detect the R Peaks
    # we can use each and check how it affects the results
    #r_peaks = detectors.hamilton_detector(FilteredData)
    #r_peaks = detectors.christov_detector(FilteredData)
    r_peaks = detectors.engzee_detector(FilteredData)
    #r_peaks = detectors.pan_tompkins_detector(FilteredData)
    #r_peaks = detectors.swt_detector(FilteredData)
    #r_peaks = detectors.two_average_detector(FilteredData)
    #r_peaks = detectors.matched_filter_detector(FilteredData)
    #r_peaks = detectors.wqrs_detector(FilteredData)
    
    
    rr = np.diff(r_peaks)
    rr_corrected = rr.copy()
    rr_corrected[np.abs(zscore(rr)) > 2] = np.median(rr)

    return rr_corrected

    
