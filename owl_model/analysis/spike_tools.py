import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from elephant.spike_train_dissimilarity import victor_purpura_distance, van_rossum_distance
from neo.core import SpikeTrain
import quantities as pq
from scipy import linalg
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline, CubicSpline, Akima1DInterpolator
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import ttest_ind


def plot_raster(spike_times, delay, duration, color='k'):

    """This function takes an array of spike times and plots the spike raster."""

    # Get the number of trials
    N_reps = len(spike_times)

    #plt.figure(figsize=(15,6))
    # Position on the parameter axis of the spikes
    for n in range(N_reps):

        # Get spike times in ms
        spikes = spike_times[n]
        
        # Plot height
        if len(np.array(spikes).reshape(-1,1)) > 0:
            y = np.ones(len(np.array(spikes).reshape(-1,1))) * n + 1

            plt.plot(spikes, y, '.', color=color, alpha=0.7, mec='w', ms=8);

            
    plt.axvspan(delay, delay+duration, facecolor='lightgray', alpha=0.15)  
    
    plt.xlabel('Time (ms)', fontsize=15)
    plt.ylabel('Trial', fontsize=15)
    plt.tick_params(labelsize=15);

    
    
def plot_raster_tuning(spike_times, parameters, delay, duration, y_label='Stimulus', color='k'):
    
    dx_percent = 0.5
    
    x = np.unique(parameters)
    
    dx = x[1] - x[0]

    #Number of unique parameter values
    N = np.size(x)
    

    for n in range(N):
        index = np.where(parameters == x[n])

        N_reps = np.size(index)
        
        
        # Get y-values for spikes at this parameter value
        y_positions = np.linspace(x[n]-dx/2*dx_percent, x[n]+dx/2*dx_percent, N_reps)
        

        for m in range(N_reps):

            # Get y-value for spikes in this trial
            y = np.ones(np.size(spike_times[index[0][m]])) * (y_positions[m])
                 

            plt.plot(spike_times[index[0][m]], y, '.', color=color, mec='w', ms=8)
            
            
            plt.ylim(x[0] - dx/2*dx_percent, x[-1] + dx/2*dx_percent)
                 
        
        # Plot a gray background over the trials at this parameter value           
        y_min = (y_positions[0] - (x[0] - dx / 2 * dx_percent) - dx*.15) / (x[-1] + dx*dx_percent - x[0])
                 
        y_max = (y_positions[-1] - (x[0] - dx / 2 * dx_percent) + dx*.15) / (x[-1] + dx*dx_percent - x[0])
        
        plt.axvspan(delay, delay+duration, ymin=y_min, ymax=y_max, facecolor='lightgray', alpha=0.35)
            
            
    plt.xlim(0, 2*delay+duration)       

    
    plt.ylabel(y_label,fontsize = 15)
    plt.xlabel('Time (ms)',fontsize = 15)
    plt.tick_params(labelsize = 13);    
    
    
def plot_tuning_curve(spike_times, parameters, x_label='Stimulus', color='b'):
    
    #spike_count = [len(spikes) for spikes in spike_times]
    spike_count = get_spike_count(spike_times)
    
    df = pd.DataFrame({'spike_count':spike_count, 'parameters':parameters})
    
    summary = df.groupby('parameters').agg(['mean', 'std'])
    
    unique_parameters = summary.index.values
    
    mean_spike_count = summary['spike_count']['mean'].values
    
    std_spike_count = summary['spike_count']['std'].values
    
    plt.errorbar(unique_parameters, mean_spike_count, std_spike_count, color=color)
    plt.plot(unique_parameters, mean_spike_count, 'o', color='k', mec='w')
    
    plt.xlabel(x_label,fontsize = 15)
    plt.ylabel('Spike count',fontsize = 15)
    plt.tick_params(labelsize = 13);     
    

def string_to_search(string, file):
    """
    Takes in string and file. Finds all lines where that string is. Returns a list of tuples.
    the tuples have the line number and the line where the string appears.

    :param string: string to search for
    :param file: location and filename to search

    :return: list of tuples, where the tuples have the line number and line where the string appears
    """

    line_number = 0
    list_of_results = []
    with open(file, 'r') as read_obj:
        for line in read_obj:
            line_number += 1
            if string in line:
                list_of_results.append((line_number, line.rstrip()))

    return list_of_results


def get_parameters_spikes(file):
    
    """
    This function extracts the stimulus parameters and spike times from an XDPHYS file
    
    Input
    file = location of a file, e.g. 'Date/1_6_06/1_6_06_data/858.01.0.itd'
    
    Output
    parameters = list of parameter values for the varied parameter for all trials
    spikes = list of spike times on each trial
    stimulus_spikes = list of spike times on each trial occuring during the stimulus
    """

    # Get the file 
    fileOpen = open(file)
    all_lines = fileOpen.readlines()
    fileOpen.close()

    # Get the stimulus parameter values ("depvar")
    list_of_results = string_to_search("depvar= ", file)

    parameters = []
    for item in list_of_results:
        string = item[1][8:]
        val = ""
        while string[0] != " ":
            val += string[0]
            string = string[1:]
        parameters.append(int(val))
        

    # Getting all the lines with "nevents=" to find the spike times
    results = string_to_search("nevents=", file)
    list_of_dur = string_to_search("Dur=", file)
    dur = int(list_of_dur[0][1][4:])*10000
    list_of_delay = string_to_search("Delay=", file)
    delay = int(list_of_delay[0][1][6:])*10000

    
    # Get the spike times
    spikes = []
    stimulus_spikes = []
    for item in results:
        number = int(item[1][8:])
        sub_list = []
        stimulus_sub_list = []
        for i in range(item[0], item[0] + number):
            string = all_lines[i]
            count = 0
            first_char = string[count]
            val = ''
            while (first_char != '-'):
                val += first_char
                count += 1
                first_char = string[count]
            val = int(val)
            sub_list.append(val/10000)
            if (val < (dur + delay)) and (val > delay):
                stimulus_sub_list.append(val/10000)
        spikes.append(sub_list)
        stimulus_spikes.append(stimulus_sub_list)
    
    
    return parameters, spikes, stimulus_spikes


def get_delay_duration(file):
    
    # Get the file 
    fileOpen = open(file)
    all_lines = fileOpen.readlines()
    fileOpen.close()
    
    # Get duraction
    list_of_dur = string_to_search("Dur=", file)
    duration = int(list_of_dur[0][1][4:])
    
    # Get delay
    list_of_delay = string_to_search("Delay=", file)
    delay = int(list_of_delay[0][1][6:])
    
    return delay, duration


  
def get_spike_count(spike_times):
    
    spike_count = np.array([])
    for spikes in spike_times:
        if isinstance(spikes, float):
            spikes = np.array(spikes).reshape(1,-1)
        spike_count = np.append(spike_count, len(spikes))
    return spike_count



def get_psth(spike_times, T_min, T_max, bin_size):
    
    number_trials = len(spike_times)
    
    # unpack all spike times into one array
    all_spikes = np.array([])
    for spikes in spike_times:
        if isinstance(spikes, float):
            spikes = np.array(spikes).reshape(1,-1)
        all_spikes = np.append(all_spikes, spikes)
      
    hist_values, bin_edges = np.histogram(all_spikes, bins=np.arange(T_min, T_max + bin_size, bin_size))
    
    return hist_values, bin_edges


    
def plot_psth(spike_times, T_min, T_max, bin_size, xlabel='Time (ms)', ylabel='Percent of trials', color='k'):
    
    number_trials = len(spike_times)
    
    # unpack all spike times into one array
    all_spikes = np.array([])
    for spikes in spike_times:
        if isinstance(spikes, float):
            spikes = np.array(spikes).reshape(1,-1)
        all_spikes = np.append(all_spikes, spikes)
      
    hist_values, bin_edges = np.histogram(all_spikes, bins=np.arange(T_min, T_max + bin_size, bin_size))
    
    plt.bar(x=bin_edges[:-1], height=hist_values / number_trials, width=np.diff(bin_edges), align='edge', color=color, alpha=0.5)
    
    plt.ylim(0,1)
    plt.tick_params(labelsize=12)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15);
    
    return hist_values, bin_edges


def plot_isih(spike_times, T_min, T_max, bin_size, xlabel='Interspike interval (ms)', ylabel='Count'):
    
    # Compute inter-spike intervals
    isi = [np.diff(spikes) for spikes in spike_times]
    
    # unpack all inter-spike intervals into one list
    all_isi = [isis for sublist in isi for isis in sublist]
    
    T_max = np.max([T_max, np.max(all_isi)])
    
    sns.histplot(all_isi, bins=np.arange(T_min, T_max + bin_size, bin_size), color='k', alpha=0.5)
           
    plt.tick_params(labelsize=12)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15);

    
    
    
# Spike time reliability

def gaussian_reliability(spike_times, sigma, T_min, T_max):
    
    """
    This function computes a correlation-based measure of spike time reliability across trials from:
    Schreiber S, Fellous JM, Whitmer D, Tiesinga P, Sejnowski TJ. A new correlation-based measure of spike timing reliability. 
    Neurocomputing (Amst). 2003 Jun 1;52-54:925-931. doi: 10.1016/S0925-2312(02)00838-X. PMID: 20740049; PMCID: PMC2926980. 
    https://pubmed.ncbi.nlm.nih.gov/20740049/
    
    Args:
        spike_times: An array of spike times for all trials
        sigma: standard deviation of the Gaussian filter applied to spike trains
        T_min: minimum spike time to include in the analysis
        T_max: maximum spike time to include in the analysis
        The units of all times should be the same
        
    Returns:
        The correlation-based measure of spike time reliability across trials
    
    """
    
    # Time for the gaussian filters
    t = np.linspace(T_min, T_max, 1000)
    
    number_trials = len(spike_times)
    
    # Initialize
    gaussian_filtered_spikes = np.zeros((number_trials, len(t)))
    
    
    # Do Gaussian filtering of all spike trains
    for i, spikes in enumerate(spike_times):
        
        if isinstance(spikes, float):
            spikes = np.array(spikes).reshape(1,-1)
    
        if len(spikes) > 0:

            for spike in np.nditer(spikes):
                gaussian_filtered_spikes[i,:] +=  np.exp(-0.5 * ((t - spike) / sigma)**2 )
                
                
    # Compute the average correlation between pairs of trials        
    R = 0
    
    for i in range(number_trials):
        
        i_norm = np.sqrt(np.dot(gaussian_filtered_spikes[i,:], gaussian_filtered_spikes[i,:]))
        
        for j in range(i+1, number_trials):          
            
            j_norm = np.sqrt(np.dot(gaussian_filtered_spikes[j,:], gaussian_filtered_spikes[j,:]))
            
            if (i_norm > 0) & (j_norm > 0):
                
                R += np.dot(gaussian_filtered_spikes[i,:], gaussian_filtered_spikes[j,:]) / (i_norm * j_norm)
    
    R = R * 2 /(number_trials * (number_trials - 1))

    return R




def gamma_factor_pair(spikes_1, spikes_2, T_min, T_max, time_bin):
    
    """
    This function compute the gamma factor measure of similarity between two spike trains. 
    See: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4042421/#B27
    
    Args:
        spikes_1: Array of spike times serving as the "reference". If data vs model, this would be the data.
        spikes_2: The other array of spike times. If data vs model, this would be the model.
        T_min: minimum spike time to include in the analysis
        T_max: maximum spike time to include in the analysis
        time_bin: time window to check for coincident spikes
        The units of all times should be the same
        
    Returns:
        gamma factor measure of similarity between two spike trains
    
    """
    
    # PSTHs of spike trains are arrays of 0's and 1's 
    psth_1, _ = get_psth(spikes_1, T_min=T_min, T_max=T_max, bin_size=time_bin)
    psth_2, _ = get_psth(spikes_2, T_min=T_min, T_max=T_max, bin_size=time_bin)
    
    if (psth_1.max() > 1) | (psth_2.max() > 1):
        print("Multiple spikes found in a time bin. Decrease the time_bin.")
    
    # Count the number of coincident spikes
    N_coinc = (psth_1 + psth_2 > 1).sum()
    
    # Number of spikes in the reference spike train
    N_reference = psth_1.sum()
    
    # Number of spikes in the comparison spike train
    N_comparison = psth_2.sum()
    
    # Firing rate of the reference spike train
    duration = T_max - T_min
    r_reference = N_reference / duration
    
    return (2 / (1 - 2 * time_bin * r_reference)) * (N_coinc - 2 * N_reference * time_bin * r_reference) / (N_reference + N_comparison)


def gamma_factor(spike_times, T_min, T_max, time_bin):
    
    """
    This function compute the gamma factor measure of similarity between spike trains. 
    This is similar to the "intrinsic gamma factor" of https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4042421/#B27
    
    Args:
        spike_times: Array of spike times 
        T_min: minimum spike time to include in the analysis
        T_max: maximum spike time to include in the analysis
        time_bin: time window to check for coincident spikes
        The units of all times should be the same
        
    Returns:
        gamma factor measure of similarity between two spike trains
    
    """
    
    number_trials = len(spike_times)
    
    gamma = np.zeros((number_trials, number_trials))
    
    for i in range(number_trials):
        for j in range(number_trials):
            
            gamma[i, j] = gamma_factor_pair(spike_times[i], spike_times[j], T_min=T_min, T_max=T_max, time_bin=time_bin)
    
    return gamma
   
    
def get_victor_purpura_distance(spike_times, cost_factor, T_min, T_max):
    
    # Convert array of spike times to neo.SpikeTrain
    spike_train = [SpikeTrain(spikes, units='ms', t_start=T_min, t_stop= T_max) for spikes in spike_times]
    
    # Compute the Victor Purpura distance for all pairs of spike trains
    victor_purpura_matrix = victor_purpura_distance(spike_train, cost_factor)
    
    # Return the mean of all unique pairs
    return np.mean(victor_purpura_matrix[np.triu_indices(victor_purpura_matrix.shape[0], k=1)])


def get_van_rossum_distance(spike_times, tau, T_min, T_max):
    
    # Convert array of spike times to neo.SpikeTrain
    spike_train = [SpikeTrain(spikes, units='ms', t_start=T_min, t_stop= T_max) for spikes in spike_times]
    
    # Compute the van rossum distance for all pairs of spike trains
    van_rossum_matrix = van_rossum_distance(spike_train, tau)
    
    # Return the mean of all unique pairs
    return np.mean(van_rossum_matrix[np.triu_indices(van_rossum_matrix.shape[0], k=1)])


def get_fano_factor(spike_times):
    
    spike_counts = get_spike_count(spike_times)

    return np.var(spike_counts, ddof=1) / np.mean(spike_counts)


## Tuning curve analysis


def tuning_curve_width(x, y, p=0.5, compute_central_width=True):
    
    """ This function finds the width of a tuning curve at p percent of the height.

    :param y: mean count list
    :param x: unique parameters

    :return: tuple with width, lower_limit, upper_limit, and the best_value
    """
    
    TOL = 0.01
    
    # Subtract off minimum
    y = y - np.min(y)

    # Rectify
    y[y < 0] = 0

    # Normalize
    y = y/np.max(y)

    # Interpolate
    xi = np.linspace(np.min(x), np.max(x), 5001)
    yi = np.interp(xi, x, y)

    # Find the best value of x at the maximum value of y
    max_index = np.argmax(yi)

    best_value = xi[max_index]

    if (max_index > 0) & (max_index < np.size(xi)-1):

        lower_x = xi[1:max_index-1]
        lower_y = yi[1:max_index-1]

        upper_x = xi[max_index+1:]
        upper_y = yi[max_index+1:]
    
        j = np.where(np.abs(lower_y - p) < TOL)
        
        if len(lower_x[j]) > 0:
         
            if compute_central_width == True:    
                lower_limit = np.max(lower_x[j])
            else:
                lower_limit = np.min(lower_x[j])
        else:
            lower_limit = np.nan
            
       
        j = np.where(np.abs(upper_y - p) < TOL)

        
        if len(upper_x[j]) > 0:    
            
            if compute_central_width == True:    
                upper_limit = np.min(upper_x[j])
            else:
                upper_limit = np.max(upper_x[j])    
            
        else:
            upper_limit = np.nan
            
    else:
        
        upper_limit = np.nan
        lower_limit = np.nan

    # Get width

    width = upper_limit - lower_limit
    
    return width, lower_limit, upper_limit, best_value
 

def fractional_energy(mean_spike_count_matrix):
    
    _, s, _ = linalg.svd(mean_spike_count_matrix, full_matrices=False)
    
    return s[0]**2 / (np.sum(s**2)) * 100



def svd_compress(A, new_rank):
    """
    This function takes a matrix A and uses the SVD to compute the best approximation of rank new_rank
    """

    if new_rank > np.linalg.matrix_rank(A):
        raise ValueError

    U, s, Vt = linalg.svd(A, full_matrices=False)
    A_s = s[0] * U[:,0].reshape(-1,1) @ Vt[0,:].reshape(1,-1)
    for k in range(1, new_rank):
        A_s += s[k] * U[:,k].reshape(-1,1) @ Vt[k,:].reshape(1,-1)
    return A_s



def two_sided_gaussian(z, mu, sigma1, sigma2, min_response1, min_response2, max_response):
    
    """ This function models ILD tuning curves with a two-sided Gaussian function with 
    variable min and max
    
    Parameters
    mu = mean of the Gaussian
    sigma1 = standard deviation of the left-side Gaussian
    sigma2 = standard deviation of the right-side Gaussian
    min_response1 = minimum response on the left size
    min_response2 = minimum response on the right size
    max_response = max response
    
    """

    r = (max_response - min_response1) * np.exp(-.5*((z - mu)/sigma1)**2) + min_response1
    r[z > mu] = (max_response - min_response2) * np.exp(-.5*((z[z > mu] - mu)/sigma2)**2) + min_response2
    
    return r

def fit_ild_curve(ILD, mean_spike_count_ild, std_spike_count_ild, plot=False):
    
    x = ILD
    y = mean_spike_count_ild
    
    
    #spl = Akima1DInterpolator(x, y) 
    
    #x_spline = np.linspace(x.min(), x.max(), 1000)
    #y_spline = spl(x_spline)
    
    
    # Interpolate
    x_spline = np.linspace(np.min(x),np.max(x),1001,endpoint=True)
    y_spline = np.interp(x_spline, x, y)
    

    # Initialize guesses for parameters (best ILD, sigma1 = 7, sigma2 = 7, min response one left, min response on right, max response)
    parameter_initial_guess = [x[np.argmax(y)], 7, 7, y[0], y[-1], np.max(y)]

    # Do the fitting
    best_parameters, _ = curve_fit(two_sided_gaussian, 
                                   xdata=x_spline, ydata=y_spline, 
                                   p0=parameter_initial_guess, 
                                   maxfev=50_000,
                                   bounds=([-40.0, 0.0, 0.0, 0.0, 0.0, 0.01], [40.0, 25.0, 25.0, np.max(y), np.max(y), np.max(y)*1.25])
                                  )

    if plot==True:
        
        #plt.figure(figsize=())
        
        x_model = np.linspace(ILD.min(), ILD.max(), 1000)

        # Get the model response with the best parameters
        model_ild_curve = two_sided_gaussian(x_model, *best_parameters) 

        # Plot the data and model
        plt.errorbar(ILD, mean_spike_count_ild, std_spike_count_ild, marker='s', label='data', alpha=0.5, color='k')
        plt.plot(x_model, model_ild_curve, label='model', color='blue')

        plt.legend()
        plt.xlabel('ILD (dB)', fontsize=15)
        plt.ylabel('Spike count', fontsize=15)
        plt.title('ILD Curve', fontsize=15);
        
        if pd.notna(best_parameters[0]):
            BV = best_parameters[0].round(2)
        
            
        plt.title('Best ILD = ' + str(BV) + ' dB', fontsize=15)

        
    return best_parameters


def get_SPS(R):
    
    # R is the ITD tuning curve
    
    R = R - np.min(R)

    if np.max(R) > 0:
        
        # Find any peaks in the ITD curve      
        index,_ = find_peaks(R, height=.1*np.max(R))
        p = R[index]
        
        # Add the end points 
        #p = np.append(p, np.array([R[0], R[-1]]))
        
        # Sort the responses from largest to smallest
        p = sorted(p, reverse=True)
        
        MP = p[0]

        if np.size(p) > 1:
            SP = p[1]
            SPS = 100 * (MP - SP) / MP
        else:
            SPS = np.nan
          
    return SPS


def fit_itd_curve(ITD, mean_spike_count_itd, std_spike_count_itd, plot=False):
    
    spl = Akima1DInterpolator(ITD, mean_spike_count_itd) 
    
    x_spline = np.linspace(ITD.min(), ITD.max(), 1000)
    y_spline = spl(x_spline)
    
    HW, LL, UL, BV = tuning_curve_width(x_spline, y_spline, p=0.5, compute_central_width=True)
    
    SPS = get_SPS(mean_spike_count_itd)
    
    if plot == True:
        plt.errorbar(ITD, mean_spike_count_itd, std_spike_count_itd, marker='s', label='data', alpha=0.5, color='k')

        plt.axvspan(LL, UL, facecolor='lightgrey', alpha=0.5)

        plt.plot(x_spline, y_spline, label='spline', color='blue')

        plt.xlabel('ITD ($\mu$s)', fontsize=15)
        plt.ylabel('Spike count', fontsize=15)
        
        if pd.notna(BV):
            BV = BV.round(2)
        if pd.notna(HW):
            HW = HW.round(2)
        if pd.notna(SPS):
            SPS = SPS.round(2)
            
        plt.title('Best ITD = ' + str(BV) + ' $\mu$s, Half-width = ' + str(HW) +  ' $\mu$s, SPS = ' + str(SPS) + '%')
        #print(BV, HW, SPS)

    return HW, LL, UL, BV, SPS



def rate_level(sound_level, A0, A1, A2, A3, A4):
    
    p = 10**(sound_level / 20)
    
    exponent = (1/(A4+1e-7)-1)
    
    d = (A3**(1 - A4)) * p/ ((A3**exponent + p**exponent)**A4)
    
    d_squared = d**2
    
    return A0 + (A1 - A0)*d_squared / (A2**2 + d_squared)



def fit_abl_curve(ABL, mean_spike_count_abl, std_spike_count_abl, plot=False):
    
    
    x = ABL
    y = mean_spike_count_abl
    
    # Interpolate
    x_spline = np.linspace(np.min(x), np.max(x), 1001, endpoint=True)
    y_spline = np.interp(x_spline, x, y)

    # Initialize guesses for parameters 
    #parameter_initial_guess = [y.min(), np.max(y) - np.min(y), 20, 25, 0.2]
    parameter_initial_guess = [y.min(), np.max(y) - np.min(y), 35, 40, 0.2]

    # Do the fitting
    #best_parameters, _ = curve_fit(rate_level, xdata=x, ydata=y, p0=parameter_initial_guess, maxfev=10000)
    best_parameters, _ = curve_fit(rate_level, 
                                   xdata=x_spline, ydata=y_spline, 
                                   p0=parameter_initial_guess, 
                                   maxfev=50_000, 
                                   bounds=([0.0, 0.0, 0.0, 0.0, 0.01], [np.nanmax(y), np.nanmax(y)*1.1, 120.0, 120.0, 0.75]))

    if plot==True:
        
        #plt.figure(figsize=())
        
        x_model = np.linspace(ABL.min(), ABL.max(), 1000)

        # Get the model response with the best parameters
        model_abl_curve = rate_level(x_model, *best_parameters) 

        # Plot the data and model
        plt.errorbar(ABL, mean_spike_count_abl, std_spike_count_abl, marker='s', label='data', alpha=0.5, color='k')
        plt.plot(x_model, model_abl_curve, label='model', color='blue')

        plt.legend()
        plt.xlabel('ABL (dB)', fontsize=15)
        plt.ylabel('Spike count', fontsize=15)
        plt.title('ABL Curve', fontsize=15);
        
    return best_parameters



def fit_frequency_curve(frequency, mean_spike_count_freq, std_spike_count_freq, plot=False):
    
    spl = Akima1DInterpolator(frequency, mean_spike_count_freq) 
    
    x_spline = np.linspace(frequency.min(), frequency.max(), 1_000)
    y_spline = spl(x_spline)
    
    # Narrowest 50% width
    HW_50, LL_50, UL_50, BV = tuning_curve_width(x_spline, y_spline, p=0.5, compute_central_width=True)
    
    # Widest 30% width
    HW_30, LL_30, UL_30, BV = tuning_curve_width(x_spline, y_spline, p=0.3, compute_central_width=False)
    
    if plot == True:
        plt.subplots(1,2, figsize=(10,4))

        plt.subplot(1,2,1)

        plt.errorbar(frequency, mean_spike_count_freq, std_spike_count_freq, marker='s', label='data', alpha=0.5, color='k')

        plt.axvspan(LL_50, UL_50, facecolor='lightgrey', alpha=0.4)

        plt.plot(x_spline, y_spline, 'b')

        plt.xlabel('Frequency (Hz)', fontsize=15)
        plt.ylabel('Spike count', fontsize=15)
        #plt.title('Narrowest 50% Width', fontsize=15);
        
        
        if pd.notna(BV):
            BV = BV.round(2)
        if pd.notna(HW_50):
            HW_50 = HW_50.round(2)
            
        plt.title('BF = ' + str(BV) + ' Hz, Width narrowest 50% = ' + str(HW_50) +  ' Hz', fontsize=12)

        

        #print(LL.round(1), UL.round(1))

        

        plt.subplot(1,2,2)
        plt.errorbar(frequency, mean_spike_count_freq, std_spike_count_freq, marker='s', label='data', alpha=0.5, color='k')

        plt.axvspan(LL_30, UL_30, facecolor='lightgrey', alpha=0.4)

        plt.plot(x_spline, y_spline, 'b')

        plt.xlabel('Frequency (Hz)', fontsize=15)
        plt.ylabel('Spike count', fontsize=15)
        #plt.title('Widest 30% Width', fontsize=15);
        
        if pd.notna(HW_30):
            HW_30 = HW_30.round(2)

        plt.title('Width widest 30% = ' + str(HW_30) +  ' Hz', fontsize=12)
              

    return HW_50, HW_30, BV


def fit_itdild_matrix(ITD, ILD, mean_spike_count_itdild, plot=False):
    
    R_1 = svd_compress(mean_spike_count_itdild, 1)
    
    frac_nrg = fractional_energy(mean_spike_count_itdild).round(3)
    
    if plot==True:


        fig = make_subplots(rows=1, cols=2,
                            specs=[[{'is_3d': True}, {'is_3d': True}]],
                            subplot_titles=['Data', 'Rank 1 approx; ' + 'frac. energy = ' + str(frac_nrg)],
                            )

        fig.add_trace(go.Surface(x=ITD, y=ILD, z = mean_spike_count_itdild, colorbar_x=-0.07), 1, 1)

        fig.add_trace(go.Surface(x=ITD, y=ILD, z=R_1), 1, 2)


        # fig.show()
        
    return fig


def fit_itd_curve_best_itd(ITD, mean_spike_count_itd, std_spike_count_itd, plot=False):
    """
    this one is for the ITD ILD analysis"""

    spl = Akima1DInterpolator(ITD, mean_spike_count_itd) 
    x_spline = np.linspace(ITD.min(), ITD.max(), 1000)
    y_spline = spl(x_spline)

    BV = x_spline[np.argmax(y_spline)]

    if plot:
        plt.errorbar(ITD, mean_spike_count_itd, std_spike_count_itd, marker='o', ms=6, color='k', alpha=0.7)
        sns.despine()
        plt.plot(x_spline, y_spline, label='spline', color='magenta')
        plt.axvline(BV, color='blue', linestyle='--', label=f'Best ITD = {BV:.2f} µs')
        plt.title(f'ITD Tuning Curve\nBest ITD = {BV:.2f} µs', fontsize=15)
        plt.xlabel('ITD (µs)', fontsize=12)
        plt.ylabel('Spike Count', fontsize=12)
        plt.legend()
        plt.show()

    return BV


def fit_itd_curve_best_itd_new(ITD, mean_spike_count_itd, std_spike_count_itd, plot=False):
    """
    Returns both the main peak ITD and a secondary peak ITD if available.
    """

    # Akima interpolation
    spl = Akima1DInterpolator(ITD, mean_spike_count_itd)
    x_spline = np.linspace(ITD.min(), ITD.max(), 1000)
    y_spline = spl(x_spline)

    # Detect peaks in the interpolated curve
    peaks, _ = find_peaks(y_spline)
    peak_itds = x_spline[peaks]
    peak_spike_counts = y_spline[peaks]

    if len(peaks) == 0:
        return None, None  # No peaks found

    # Sort peaks by spike count (descending order)
    sorted_indices = np.argsort(peak_spike_counts)[::-1]
    sorted_peak_itds = peak_itds[sorted_indices]
    sorted_peak_spike_counts = peak_spike_counts[sorted_indices]

    # Identify the main peak (highest spike count)
    main_peak_itd = sorted_peak_itds[0]
    main_peak_spike_count = sorted_peak_spike_counts[0]

    # Identify a secondary peak (if available and ≥50% of the main peak)
    secondary_peak_itd = None
    for i in range(1, len(sorted_peak_itds)):
        if sorted_peak_spike_counts[i] >= 0.5 * main_peak_spike_count:
            secondary_peak_itd = sorted_peak_itds[i]
            break

    if plot:
        plt.figure(figsize=(8, 5))
        plt.errorbar(ITD, mean_spike_count_itd, std_spike_count_itd, marker='o', ms=6, color='k', alpha=0.7)
        sns.despine()
        plt.plot(x_spline, y_spline, label='Spline Interpolation', color='magenta')
        plt.axvline(main_peak_itd, color='blue', linestyle='--', label=f'Main Peak ITD = {main_peak_itd:.2f} µs')

        if secondary_peak_itd is not None:
            plt.axvline(secondary_peak_itd, color='red', linestyle='--', label=f'Secondary Peak ITD = {secondary_peak_itd:.2f} µs')

        plt.title(f'ITD Tuning Curve\nMain Peak = {main_peak_itd:.2f} µs', fontsize=15)
        plt.xlabel('ITD (µs)', fontsize=12)
        plt.ylabel('Spike Count', fontsize=12)
        plt.legend()
        plt.show()

    return main_peak_itd, secondary_peak_itd


def fit_itd_curve_3_peaks(ITD, mean_spike_count_itd, std_spike_count_itd, plot=False):
    """
    Returns the main peak ITD, secondary peak ITD, and third peak ITD if available.
    """

    # Akima interpolation
    spl = Akima1DInterpolator(ITD, mean_spike_count_itd)
    x_spline = np.linspace(ITD.min(), ITD.max(), 1000)
    y_spline = spl(x_spline)

    # Detect peaks in the interpolated curve
    peaks, _ = find_peaks(y_spline)
    peak_itds = x_spline[peaks]
    peak_spike_counts = y_spline[peaks]

    if len(peaks) == 0:
        return None, None, None  # No peaks found

    # Sort peaks by spike count (descending order)
    sorted_indices = np.argsort(peak_spike_counts)[::-1]
    sorted_peak_itds = peak_itds[sorted_indices]
    sorted_peak_spike_counts = peak_spike_counts[sorted_indices]

    # Identify the main peak (highest spike count)
    main_peak_itd = sorted_peak_itds[0]
    main_peak_spike_count = sorted_peak_spike_counts[0]

    # Identify a secondary peak (if available and ≥50% of the main peak)
    secondary_peak_itd = None
    third_peak_itd = None

    valid_peaks = [
        (itd, count) for itd, count in zip(sorted_peak_itds[1:], sorted_peak_spike_counts[1:])
        if count >= 0.5 * main_peak_spike_count
    ]

    # Assign secondary and third peak if available
    if len(valid_peaks) >= 1:
        secondary_peak_itd = valid_peaks[0][0]  # Second highest peak
    if len(valid_peaks) >= 2:
        third_peak_itd = valid_peaks[1][0]  # Third highest peak

    # Plot the curve if requested
    if plot:
        plt.figure(figsize=(8, 5))
        plt.errorbar(ITD, mean_spike_count_itd, std_spike_count_itd, marker='o', ms=6, color='k', alpha=0.7)
        sns.despine()
        plt.plot(x_spline, y_spline, label='Spline Interpolation', color='magenta')
        plt.axvline(main_peak_itd, color='blue', linestyle='--', label=f'Main Peak ITD = {main_peak_itd:.2f} µs')

        if secondary_peak_itd is not None:
            plt.axvline(secondary_peak_itd, color='red', linestyle='--', label=f'Secondary Peak ITD = {secondary_peak_itd:.2f} µs')
        if third_peak_itd is not None:
            plt.axvline(third_peak_itd, color='green', linestyle='--', label=f'Third Peak ITD = {third_peak_itd:.2f} µs')

        plt.title(f'ITD Tuning Curve\nMain Peak = {main_peak_itd:.2f} µs', fontsize=15)
        plt.xlabel('ITD (µs)', fontsize=12)
        plt.ylabel('Spike Count', fontsize=12)
        plt.legend()
        plt.show()

    return main_peak_itd, secondary_peak_itd, third_peak_itd
