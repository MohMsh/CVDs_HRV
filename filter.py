from scipy.signal import butter, iirnotch, lfilter

cutoff_high = 0.5
cutoff_low = 2
powerline = 60
order = 5


## FIR Filters: Also known as the Windowing or Band-pass filters
## A high pass filter allows frequencies higher than a cut-off value
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False, output='ba')
    return b, a
## A low pass filter allows frequencies lower than a cut-off value
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False, output='ba')
    return b, a

## =================================================================================

## IIR Notch Filters: Usually know as Notch filters    
## Used to remove power line interference and/or a motion artifact
def notch_filter(cutoff, fs, q):
    nyq = 0.5*fs
    freq = cutoff/nyq
    b, a = iirnotch(freq, q)
    return b, a
    
## ==================================================================================

## The main filtering function   
def final_filter(data, fs, order=5):
    b, a = butter_highpass(cutoff_high, fs, order=order)
    x = lfilter(b, a, data)
    d, c = butter_lowpass(cutoff_low, fs, order = order)
    y = lfilter(d, c, x)
    f, e = notch_filter(powerline, fs, 30)
    z = lfilter(f, e, y)
    return z
