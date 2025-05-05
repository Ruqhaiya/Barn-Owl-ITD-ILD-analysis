import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import f_oneway


def get_file_name_mat(df, neuron_index, file_type, location='/Users/fischer9/Dropbox/MATLAB/OwlData/LS'):
    
    """
    This function gets the file name for a .mat file of spike counts
    """
    
    month = str(df.loc[neuron_index, 'month'])
    day = str(df.loc[neuron_index, 'day'])
    year = str(df.loc[neuron_index, 'year'])
    owl = str(df.loc[neuron_index, 'owl'])
    neuron = '0' + str(df.loc[neuron_index, 'neuron']) if df.loc[neuron_index, 'neuron'] < 10 else str(df.loc[neuron_index, 'neuron'])
    order = str(df.loc[neuron_index, 'order']) 

    date = month + '_' + day + '_0' + year

    #return location + '/Date/' + date + '/' + date + '_data/' + owl + '.' + neuron + '.' + order + '.' + file_type # Use this to load XDPHYS files
    return location + '/Date/' + date + '/Rates/r' + date + '_'+ owl + '_' + neuron + '_' + order + '_' + file_type + '.mat'


def load_itdild_mat(filename):
    
    """
    Load ITD, ILD spike count response data from a MATLAB file.

    Parameters
    ----------
    filename : str
        The name of the MATLAB file to load.

    Returns
    -------
    ITD : array
        The interaural time difference (ITD) values in microseconds.
    ILD : array
        The interaural level difference (ILD) values in decibels.
    mean_spike_count : array
        The mean spike count per stimulus across trials.
    std_spike_count : array
        The standard deviation of the spike count per stimulus across trials.
    """
    
    
    mat = sio.loadmat(filename, squeeze_me=True)
    
    ITD = mat['x']

    ILD = mat['y']

    mean_spike_count = mat['mM']

    std_spike_count = mat['sM']
    
    return ITD, ILD, mean_spike_count, std_spike_count


def load_freq_mat(filename):
    
    mat = sio.loadmat(filename, squeeze_me=True)
    
    frequency = mat['x']

    mean_spike_count = mat['mM']

    std_spike_count = mat['sM']
    
    # trial_spike_count = mat['rawM']
    
    # F, p = f_oneway(*trial_spike_count.T)
    
    return frequency, mean_spike_count, std_spike_count


def load_itd_mat(filename):
    
    mat = sio.loadmat(filename, squeeze_me=True)
    
    ITD = mat['x']

    mean_spike_count = mat['mM']

    std_spike_count = mat['sM']
    
    return ITD, mean_spike_count, std_spike_count

def load_ild_mat(filename):
    
    mat = sio.loadmat(filename, squeeze_me=True)
    
    ILD = mat['x']

    mean_spike_count = mat['mM']

    std_spike_count = mat['sM']
    
    return ILD, mean_spike_count, std_spike_count

def load_frozen_itdild(filename):
    
    mat = sio.loadmat(filename, squeeze_me=True)
    
    spike_times = mat['spike_times'][2]
    spike_times_new = np.array([subarray[0] for subarray in spike_times if subarray.size > 0], dtype=object)

    return spike_times



def load_abl_mat(filename):
    
    mat = sio.loadmat(filename, squeeze_me=True)
    
    ABL = mat['x']

    mean_spike_count = mat['mM'][0]

    std_spike_count = mat['sM'][0]
    
    return ABL, mean_spike_count, std_spike_count



def get_itd_ild_curves(ITD, ILD, mean_spike_count_itdild, std_spike_count_itdild):
    
    """Computes the best ITD and ILD curves from the given spike count data.

    Args:
        ITD: A numpy array of shape (n,) containing the ITD values.
        ILD: A numpy array of shape (m,) containing the ILD values.
        mean_spike_count_itdild: A numpy array of shape (m, n) containing the mean spike counts for each combination of ITD and ILD.
        std_spike_count_itdild: A numpy array of shape (m, n) containing the standard deviation of spike counts for each combination of ITD and ILD.

    Returns:
        A tuple of four numpy arrays:
        - mean_spike_count_itd: A numpy array of shape (n,) containing the mean spike counts for the best ITD curve.
        - std_spike_count_itd: A numpy array of shape (n,) containing the standard deviation of spike counts for the best ITD curve.
        - mean_spike_count_ild: A numpy array of shape (m,) containing the mean spike counts for the best ILD curve.
        - std_spike_count_ild: A numpy array of shape (m,) containing the standard deviation of spike counts for the best ILD curve.
    """
    
    best_ild_index, best_itd_index = np.unravel_index(mean_spike_count_itdild.argmax(), mean_spike_count_itdild.shape)

    mean_spike_count_itd = mean_spike_count_itdild[best_ild_index, :]
    
    std_spike_count_itd = std_spike_count_itdild[best_ild_index, :]
    
    mean_spike_count_ild = mean_spike_count_itdild[:, best_itd_index]
    
    std_spike_count_ild = std_spike_count_itdild[:, best_itd_index]

    return mean_spike_count_itd, std_spike_count_itd, mean_spike_count_ild, std_spike_count_ild
