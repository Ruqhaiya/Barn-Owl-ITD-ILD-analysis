import numpy as np

"""
These files need updating to handle multiple neurons with
different parameters. Include code from copied folder.
\
"""

def two_sided_gaussian(z, mu, sigma1, sigma2):

    r = np.exp(-.5*((z - mu)/sigma1)**2)
    r[z > mu] = np.exp(-.5*((z[z > mu] - mu)/sigma2)**2)

    return r



def iccl_input(ICcl, e_contra, x, z):

    #input_ild = e_contra * two_sided_gaussian(z, ICcl.best_ild, ICcl.ild_sigma_1, ICcl.ild_sigma_2)
    
    iccl_input = np.zeros((ICcl.number_neurons, len(z)))
    
    for n in range(ICcl.number_neurons):
    
        input_ild = two_sided_gaussian(z, ICcl.best_ild[n], ICcl.ild_sigma_1[n], ICcl.ild_sigma_2[n])

        itd_index = np.argmin(np.abs(ICcl.internal_ITD - ICcl.best_itd[n]))

        input_itd = np.squeeze(x[itd_index, :])

        #iccl_input =  ICcl.adexlif_parameters['input_gain'] * (input_ild + input_itd) + ICcl.adexlif_parameters['input_bias']

        iccl_input[n,:] =  ICcl.adexlif_parameters['input_gain'][n] * (input_ild * input_itd) + ICcl.adexlif_parameters['input_bias'][n]

    return iccl_input


def get_iccl_adexlif_spikes(I, dt, adexlif_parameters, record_potentials = True):

    """
    This function generates spikes for a population of ICcl neurons using
    the adaptive exponential integrate and fire neuron (Romain reference)

    Inputs
    I : Input currents for each neuron formed from the front end cue extration model (numpy array of float)
    dt: time step in ms (float)
    adexlif_parameters: a dictionary of the neuron parameters
    record_potentials : indicator of whether to record membrane potentials (bool)

    Output
    spike_counts : spike count for each neuron
    spike_times : spike times for each neuron
    membrane_potentials
    """

    # Neuron parameters
    Tref = adexlif_parameters['Tref']
    Vrest = adexlif_parameters['Vrest']
    Vreset = adexlif_parameters['Vreset']
    EL = adexlif_parameters['EL']
    tRC = adexlif_parameters['tRC']
    #C = adexlif_parameters['C']

    #Adaptive threshold parameters
    Del = adexlif_parameters['Del']
    th_range = adexlif_parameters['th_range']
    tau_th = adexlif_parameters['tau_th']
    alpha = adexlif_parameters['alpha']
    ka = adexlif_parameters['ka']
    VT = adexlif_parameters['VT']
    Vi = adexlif_parameters['Vi']
    ki = adexlif_parameters['ki']
    MaxFR = 1/Tref
    eps = dt/tRC


    # Get population size and number of time steps in input signal
    if I.ndim == 1:
        number_neurons, number_time = 1, len(I)
        I = I.reshape(1,-1)
    elif I.ndim > 1:
        number_neurons, number_time = I.shape

    # Time
    t = np.arange(0, number_time) * dt

    # Intialize
    in_refractory = np.zeros(number_neurons)

    spike_counts = np.zeros(number_neurons)

    spike_times = np.zeros((number_neurons, int(np.floor(dt*number_time*MaxFR)+2))) # max possible # spikes.

    V = Vrest + np.random.uniform(-5, 5, number_neurons)

    Vthres = alpha*(V - Vi) + VT + ka* np.log(1 + np.exp((V - Vi)/ki))

    # Record potential for plotting, if desired
    if record_potentials == True:
        Vm = np.ones((number_neurons, number_time))*Vreset

        Vm[:,0] = V

    # Run the simulation
    for n in range(number_time-1):

        # Update threshold
        th_inf = alpha*(V - Vi) + VT + ka*np.log(1 + np.exp((V - Vi)/ki))

        Vthres = Vthres + dt/tau_th*(th_inf - Vthres)

        spike_threshold = Vthres + np.random.uniform(-0.5, 0.5, number_neurons)*th_range + 3

        # Membrane potential update
        dV = (EL-V) + Del*np.exp((V-VT)/Del) + I[:,n]

        # If out of refractory period, update membrane potential
        V[in_refractory <= 0] += eps*dV[in_refractory <= 0]

        if record_potentials == True:
            Vm[in_refractory <= 0, n+1] = V[in_refractory <= 0]

        # Find spiking neurons, where the potential V exceeds the spike threshold
        spiking_neurons = V > spike_threshold

        # Increase the spike counts
        spike_counts[spiking_neurons] += 1

        # Record the spike times
        spike_times[spiking_neurons, spike_counts[spiking_neurons].astype(int)-1] = t[n]

        # Reset the refactory period counter
        in_refractory[spiking_neurons] = Tref

        # Reset the membrane potential to the reset value
        V[spiking_neurons] = Vreset

        if record_potentials == True:
            # Artificial spike height for plotting
            Vm[spiking_neurons, n+1] = -10

        # Decrease the refactory period counter for those in the refractory period
        in_refractory[in_refractory > 0] -= dt

    spike_times = spike_times.squeeze()

    if number_neurons == 1:
        spike_times = spike_times[:spike_counts.squeeze().astype(int)]
    else:
        spike_times = [spike_times[n, :spike_counts[n].squeeze().astype(int)] for n in range(len(spike_counts))]

    if record_potentials == True:
        Vm[Vm > -10] = -10
        return Vm, spike_counts, spike_times
    else:
        return spike_counts, spike_times

    
    
    
def get_iccl_adexlif_spikes_jitter(I, dt, adexlif_parameters, record_potentials = True):

    """
    This function generates spikes for a population of ICcl neurons using
    the adaptive exponential integrate and fire neuron (Romain reference)

    Inputs
    I : Input currents for each neuron formed from the front end cue extration model (numpy array of float)
    dt: time step in ms (float)
    adexlif_parameters: a dictionary of the neuron parameters
    record_potentials : indicator of whether to record membrane potentials (bool)

    Output
    spike_counts : spike count for each neuron
    spike_times : spike times for each neuron
    membrane_potentials
    """
    
    # Threshold noise
    jitter_range = adexlif_parameters['jitter_range']
    spike_probability = adexlif_parameters['spike_probability']

    # Neuron parameters
    Tref = adexlif_parameters['Tref']
    Vrest = adexlif_parameters['Vrest']
    Vreset = adexlif_parameters['Vreset']
    EL = adexlif_parameters['EL']
    tRC = adexlif_parameters['tRC']
    #C = adexlif_parameters['C']

    #Adaptive threshold parameters
    Del = adexlif_parameters['Del']
    th_range = adexlif_parameters['th_range']
    tau_th = adexlif_parameters['tau_th']
    alpha = adexlif_parameters['alpha']
    ka = adexlif_parameters['ka']
    VT = adexlif_parameters['VT']
    Vi = adexlif_parameters['Vi']
    ki = adexlif_parameters['ki']
    MaxFR = 1/Tref
    eps = dt/tRC


    # Get population size and number of time steps in input signal
    if I.ndim == 1:
        number_neurons, number_time = 1, len(I)
        I = I.reshape(1,-1)
    elif I.ndim > 1:
        number_neurons, number_time = I.shape

    # Time
    t = np.arange(0, number_time) * dt

    # Intialize
    in_refractory = np.zeros(number_neurons)

    spike_counts = np.zeros(number_neurons)

    spike_times = np.zeros((number_neurons, int(np.floor(dt*number_time*MaxFR)+2))) # max possible # spikes.

    #V = Vrest + np.random.uniform(-5, 5, number_neurons)
    
    V = np.ones(number_neurons)*Vrest 

    Vthres = alpha*(V - Vi) + VT + ka* np.log(1 + np.exp((V - Vi)/ki))

    # Record potential for plotting, if desired
    if record_potentials == True:
        Vm = np.ones((number_neurons, number_time))*Vreset

        Vm[:,0] = V
        
        dVm = np.zeros((number_neurons, number_time))

    # Run the simulation
    for n in range(number_time-1):

        # Update threshold
        th_inf = alpha*(V - Vi) + VT + ka*np.log(1 + np.exp((V - Vi)/ki))

        Vthres = Vthres + dt/tau_th*(th_inf - Vthres)       

        # Membrane potential update
        dV = (EL - V) + Del*np.exp((V - VT)/Del) + I[:,n]
        
        voltage_derivative = dV/tRC
        
        

        # If out of refractory period, update membrane potential
        V[in_refractory <= 0] += eps*dV[in_refractory <= 0]

        if record_potentials == True:
            Vm[in_refractory <= 0, n+1] = V[in_refractory <= 0]
            
            dVm[:,n] = voltage_derivative
            
            
        # Spike threshold. Add 3 mV and shift threshold based on voltage derivative as in Fontaine et al. 
        spike_threshold = Vthres + 3 - 0.71 * voltage_derivative
        

        # Find spiking neurons, where the potential V exceeds the spike threshold          
        spiking_neurons = V > spike_threshold 
        
        # Randomly select spikes to keep
        spike_success = (V > spike_threshold) & (np.random.uniform(0, 1, number_neurons) < spike_probability)
        
        # Increase the spike counts
        spike_counts[spike_success] += 1

        # Record the spike times with jitter added to the spike time        
        spike_times[spike_success, spike_counts[spike_success].astype(int)-1] = t[n] + np.random.uniform(-0.5, 0.5, sum(spike_success))*jitter_range

        # Reset the refactory period counter
        in_refractory[spiking_neurons] = Tref
 
        # Reset the membrane potential to the reset value
        V[spiking_neurons] = Vreset
 
        if record_potentials == True:
            # Artificial spike height for plotting
            Vm[spike_success, n+1] = -10

        # Decrease the refactory period counter for those in the refractory period
        in_refractory[in_refractory > 0] -= dt

    # Convert spike times to list         
    spike_times = spike_times.squeeze()

    if number_neurons == 1:
        spike_times = spike_times[:spike_counts.squeeze().astype(int)]
    else:
        spike_times = [spike_times[n, :spike_counts[n].squeeze().astype(int)] for n in range(len(spike_counts))]

    if record_potentials == True:
        Vm[Vm > -45] = -10
        return Vm, spike_counts, spike_times, dVm
    else:
        return spike_counts, spike_times
