import numpy as np

from owl_model.model import front_end




# THis class is not done. Need to generate parameters for 
class ICcl:
    
    def __init__(self, number_frequencies, neurons_per_frequency):
        
         
        self.number_frequencies = number_frequencies
        self.neurons_per_frequency = neurons_per_frequency
           
        self.best_frequencies = np.linspace(0.5, 10, number_frequencies).reshape(-1,1) * np.ones(neurons_per_frequency)
        self.best_itds = np.tile(np.linspace(-250, 250, neurons_per_frequency), (number_frequencies,1))

        ## Fixed response parameters      
        self.TAU_ENERGY = 2

        self.GAMMA = 100

        self.XCORR_BIAS = 1
        self.XCORR_GAIN_SCALE = 2.65 / 6
        self.XCORR_GAIN_BIAS = 15 / 6
        self.XCORR_GAIN_TAU = 3
        self.XCORR_TAU = 5
        
        self.R_MAX = 10
        
        self.adexlif_parameters = {"Tref" : 1, 
                           "Vrest" : -70,
                           "Vreset" : -55,
                           "EL" : -75,
                           "tRC" : 60,
                           "C" : 5,
                           "th_range" : 0,
                           "Del" : 1,
                           "tau_th" : 5,
                           "alpha" : 0,
                           "ka" : 5,
                           "VT" : -50,
                           "Vi" : -60,
                           "ki" : 1,
                           'jitter_range' : 0.5,
                           'spike_probability' : 0.9}
        
        
class ICcl_neuron:
    
    def __init__(self, best_frequency, best_itd, best_ild, ild_sigma_1, ild_sigma_2, brain_side, sampling_rate):
              
        self.best_frequency = best_frequency
        self.best_itd = best_itd
        self.best_ild = best_ild
        self.ild_sigma_1 = ild_sigma_1
        self.ild_sigma_2 = ild_sigma_2
        self.brain_side = brain_side
           
        self.sampling_rate = sampling_rate
        
        ## Fixed response parameters      
        self.TAU_ENERGY = 1
        
        self.TAU_GAIN = 1
        
        self.TAU_ENVELOPE = 2

        self.GAMMA = 100

        self.XCORR_BIAS = 0.1 #old 1
        self.XCORR_GAIN_SCALE = 2.65 / 6
        self.XCORR_GAIN_BIAS = 15 / 6
        self.XCORR_GAIN_TAU = 15 # was 3
        self.XCORR_TAU = 3
        
        self.delays = np.arange(0, 0.3005, 0.005).reshape(-1,1)
        self.internal_ITD = (self.delays[:,0] - np.flipud(self.delays[:,0])) * 1000
        
        self.R_MAX = 10
        
        self.filter_coefs = front_end.getFilterCoefs(np.array([best_frequency*1000]), sampling_rate)
        
        self.adexlif_parameters = {"Tref" : 0.8, 
                           "Vrest" : -70,
                           "Vreset" : -70,
                           "EL" : -75,
                           "tRC" : 5,
                           "C" : 5,
                           "th_range" : 0,
                           "Del" : 1,
                           "tau_th" : 5,
                           "alpha" : 0,
                           "ka" : 5,
                           "VT" : -63,
                           "Vi" : -67,
                           "ki" : 1,
                           "input_gain" : 8,
                           "input_bias" : -7.5,
                           'jitter_range' : 0.5,
                           'spike_probability' : 0.9}
   

