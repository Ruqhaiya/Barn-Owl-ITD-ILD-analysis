import numpy as np
from scipy.interpolate import interp1d



def gen_bandpass_psd(w_min, w_max, w):

    """This function generates the power spectral density of the signal"""

    Nf = len(w) # number of frequencies in signal
    power = np.zeros(Nf)

    power[(np.abs(w) <= w_max) & (np.abs(w) > w_min)] = 1
    
    return power


def gen_bandpass_signal(T, dt, f_min, f_max):
    """ This function generates a random bandpass signal"""

    # Generate frequencies
    Nt = 2 * int(T / dt / 2) + 1 # Number of time samples of s
    Nf = Nt # # samples doesn't change when converting to freq domain!
    df = 1 / T # Hz
    f = np.arange(-(Nt - 1) / 2, (Nt - 1) / 2 + 1) * df # Hz
    #w = 2 * np.pi * f # rad/sec

    #S = np.zeros(Nt)
    
    # Generate the power spectrum of the signal
    power = np.zeros(Nf)
    power[(np.abs(f) <= f_max) & (np.abs(f) > f_min)] = 1

    # Generate the signal
    #power = gen_bandpass_psd(w_min, w_max, w)

    S = np.sqrt(power / 2) * (np.random.randn(Nt) + 1j * np.random.randn(Nt)) # generate the 'Amplitudes' (signal in the Fourier domain)

    s = np.real(np.fft.ifft(np.fft.ifftshift(S))) # transform to time domain
    s = s - np.mean(s) # set the mean to zero. (Note that this changes the power spectrum-cha)

    # Apply an onset/offset ramp of 5 ms
    ramp = np.ones(s.shape)

    N = int(5/dt)
    t = np.arange(Nt) * dt

    ramp[:N] = t[:N]/t[N]
    ramp[-N:] = 1 + (t[-N:] - t[-N])/(t[-N] - t[-1])

    s = s * ramp

    # Set the RMS to 1
    s = s / np.sqrt(np.mean(s**2)) 
    
    return s


def gen_noise_inputs(ITD, ILD, ABL, T, time_step, f_min, f_max):

    # Generate a sound signal
    sound = gen_bandpass_signal(T + 2, time_step, f_min, f_max)
    
    # Set rms values to produce desired ABL and ILD
    RMSL = ABL - .5*ILD
    RMSR = ABL + .5*ILD

    rmsL = 2*10**(RMSL/20)
    rmsR = 2*10**(RMSR/20)
    
    # Use interpolation to shift signal to produce desired ITD
    t_full = np.arange(len(sound))*time_step

    t = t_full[t_full <= T]

    f = interp1d(t_full, sound, kind = 'cubic', fill_value="extrapolate")

    if ITD >= 0:
        sL = rmsL * f(t)
        sR = rmsR * f(t + ITD/1000)
    else:
        sL = rmsL * f(t - ITD/1000)
        sR = rmsR * f(t)

    return sL, sR



def gen_inputs_bc(ITD, ILD, ABL, BC, T, time_step, f_min, f_max):

    # Generate the sound signal
    sound = gen_bandpass_signal(T + 2, time_step, f_min, f_max)
      
    # Use interpolation to shift signal to produce desired ITD
    t_full = np.arange(len(sound)) * time_step

    t = t_full[t_full <= T]

    f = interp1d(t_full, sound, kind = 'cubic', fill_value="extrapolate")

    if ITD >= 0:
        signal_left = f(t)
        signal_right = f(t + ITD/1000)
    else:
        signal_left = f(t - ITD/1000)
        signal_right = f(t)
        
    # Add uncorrelated noise to the correlated signal    
    noise_left = gen_bandpass_signal(T, time_step, f_min, f_max)
    noise_right = gen_bandpass_signal(T, time_step, f_min, f_max)

    noise_ratio = np.sqrt(1/BC-1)
    
    sL = signal_left + noise_left*noise_ratio

    sR = signal_right + noise_right*noise_ratio
    
    
    # Set rms values to produce desired ABL and ILD
    RMSL = ABL - .5*ILD
    RMSR = ABL + .5*ILD

    rmsL = 2*10**(RMSL/20)
    rmsR = 2*10**(RMSR/20)

    sL = sL / np.sqrt( np.mean(sL**2) ) * rmsL

    sR = sR / np.sqrt( np.mean(sR**2) ) * rmsR

    return sL, sR


def gen_tone_inputs(ITD, ILD, ABL, T, time_step, frequency):

    # Generate a sound signal
    #sound = gen_bandpass_signal(T + 2, time_step, f_min, f_max)
    t = np.arange(0, T+time_step, time_step)
    
    # Set rms values to produce desired ABL and ILD
    RMSL = ABL - .5*ILD
    RMSR = ABL + .5*ILD

    rmsL = 2*10**(RMSL/20)
    rmsR = 2*10**(RMSR/20)
    
    sL = rmsL * np.sin(2 * np.pi * frequency * t)
    sR = rmsR * np.sin(2 * np.pi * frequency * (t + ITD/1000))
    
        
    # Apply an onset/offset ramp of 5 ms
    ramp = np.ones(sL.shape)

    N = int(5/time_step)

    ramp[:N] = t[:N]/t[N]
    ramp[-N:] = 1 + (t[-N:] - t[-N])/(t[-N] - t[-1])

    sL = sL * ramp
    sR = sR * ramp

    return sL, sR
