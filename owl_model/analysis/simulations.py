import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from owl_model.model import stimulus, front_end, iccl_group, network


def get_cues(ICcl, ITD, ILD, ABL, T, time_step, f_min, f_max): 
    
    sL, sR = stimulus.gen_noise_inputs(ITD, ILD, ABL, T, time_step, f_min, f_max)

    #cochlea_left = front_end.ERBFilterBank(sL, ICcl.filter_coefs)

    #cochlea_right = front_end.ERBFilterBank(sR, ICcl.filter_coefs)

    #e_left, e_right, z, x = front_end.iccl_inputs(ICcl, cochlea_left, cochlea_right, T, time_step)
    
    e_left, e_right, z, x = front_end.get_iccl_inputs(ICcl, sL, sR, T, time_step)
    
    return e_left, e_right, z, x
 

def get_plot_cues(ICcl, ITD, ILD, ABL, T, time_step, f_min, f_max):
    
    e_left, e_right, z, x = get_cues(ICcl, ITD, ILD, ABL, T, time_step, f_min, f_max)
    
    t = np.arange(0, T+time_step, time_step) 
    
    plt.subplots(1, 3, figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.contourf(t, ICcl.internal_ITD, x)
    plt.colorbar()
    plt.xlabel('Time (ms)', fontsize=15)
    plt.ylabel('Internal best ITD ($\mu$s)', fontsize=15)
    plt.title('Running cross-correlation', fontsize=15)
    
    plt.subplot(1, 3, 2)
    plt.plot(t, e_left)
    plt.plot(t, e_right)
    plt.legend(['left', 'right'])
    plt.xlabel('Time (ms)', fontsize=15)
    plt.ylabel('Energy', fontsize=15)
    plt.title('Monaural energy', fontsize=15)
    
    plt.subplot(1, 3, 3)
    plt.plot(t, z)
    plt.xlabel('Time (ms)', fontsize=15)
    plt.ylabel('z', fontsize=15)
    plt.title('Log energy difference', fontsize=15)
    
    plt.subplots_adjust(hspace=0.5)
    
    
   
def iccl_spike_response(ICcl, ITD, ILD, ABL, T, time_step, f_min, f_max):
       
    e_left, e_right, z, x = get_cues(ICcl, ITD, ILD, ABL, T, time_step, f_min, f_max)

    if ICcl.brain_side == 'L':
        I = network.iccl_input(ICcl, e_right, x, z)
    elif ICcl.brain_side == 'R':
        I = network.iccl_input(ICcl, e_left, x, z)

    Vm, spike_counts, spike_times, _ = network.get_iccl_adexlif_spikes_jitter(I, 
                                                                      time_step, 
                                                                      ICcl.adexlif_parameters, 
                                                                      record_potentials = True)

    t = np.arange(0, T+time_step, time_step)
    
    for n in range(Vm.shape[0]):
        plt.plot(t, Vm[n,:])
    
    plt.xlabel('Time (ms)', fontsize=15)
    plt.ylabel('Membrane potential (mV)', fontsize=15)
    
    return Vm, spike_counts, spike_times


def iccl_spike_response_frozen(ICcl, ITD, ILD, ABL, T, time_step, f_min, f_max, number_trials):
       
    e_left, e_right, z, x = get_cues(ICcl, ITD, ILD, ABL, T, time_step, f_min, f_max)

    if ICcl.brain_side == 'L':
        I = network.iccl_input(ICcl, e_right, x, z)
    elif ICcl.brain_side == 'R':
        I = network.iccl_input(ICcl, e_left, x, z)
        
    I = np.tile(I, (number_trials,1))

    Vm, spike_counts, spike_times, _ = network.get_iccl_adexlif_spikes_jitter(I, 
                                                                      time_step, 
                                                                      ICcl.adexlif_parameters, 
                                                                      record_potentials = True)

    t = np.arange(0, T+time_step, time_step)
    
    for n in range(Vm.shape[0]):
        plt.plot(t, Vm[n,:])
    
    plt.xlabel('Time (ms)', fontsize=15)
    plt.ylabel('Membrane potential (mV)', fontsize=15)
    
    return Vm, spike_counts, spike_times
   
    
    
def itd_curve_iccl(ICcl, ITDs, number_trials, ILD, ABL, T, time_step, f_min, f_max):
    
    spike_counts = np.zeros((number_trials, len(ITDs)))

    for i in range(number_trials):

        for j, ITD in enumerate(ITDs):

            e_left, e_right, z, x = get_cues(ICcl, ITD, ILD, ABL, T, time_step, f_min, f_max)
            
            if ICcl.brain_side == 'L':
                I = network.iccl_input(ICcl, e_right, x, z)
            elif ICcl.brain_side == 'R':
                I = network.iccl_input(ICcl, e_left, x, z)

            spike_counts[i, j], spike_times = network.get_iccl_adexlif_spikes_jitter(I, 
                                                                              time_step, 
                                                                              ICcl.adexlif_parameters, 
                                                                              record_potentials = False)

    
    mean_count = np.mean(spike_counts, axis=0)
    sd_count = np.std(spike_counts, axis=0, ddof=1)

    plt.errorbar(ITDs, mean_count, sd_count)
    
    plt.xlabel('ITD ($\mu$s)', fontsize=15)
    plt.ylabel('Number of spikes', fontsize=15)
    plt.title('ITD tuning curve', fontsize=15)
    
    
    
def ild_curve_iccl(ICcl, ILDs, number_trials, ITD, ABL, T, time_step, f_min, f_max):
    
    spike_counts = np.zeros((number_trials, len(ILDs)))

    for i in range(number_trials):

        for j, ILD in enumerate(ILDs):
            
            e_left, e_right, z, x = get_cues(ICcl, ITD, ILD, ABL, T, time_step, f_min, f_max)

            if ICcl.brain_side == 'L':
                I = network.iccl_input(ICcl, e_right, x, z)
            elif ICcl.brain_side == 'R':
                I = network.iccl_input(ICcl, e_left, x, z)


            spike_counts[i, j], spike_times = network.get_iccl_adexlif_spikes_jitter(I, 
                                                                              time_step, 
                                                                              ICcl.adexlif_parameters, 
                                                                              record_potentials = False)

    
    mean_count = np.mean(spike_counts, axis=0)
    sd_count = np.std(spike_counts, axis=0, ddof=1)

    plt.errorbar(ILDs, mean_count, sd_count)
    
    plt.xlabel('ILD (dB)', fontsize=15)
    plt.ylabel('Number of spikes', fontsize=15)
    plt.title('ILD tuning curve', fontsize=15)