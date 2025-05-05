import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.interpolate import Akima1DInterpolator
from scipy.special import expit

def MakeERBFilters(fs, cf, lowFreq, Q10):

    """
    Original Matlab comments:
    % 
    % This function computes the filter coefficients for a bank of 
    % Gammatone filters.  These filters were defined by Patterson and 
    % Holdworth for simulating the cochlea.  
    % 
    % The result is returned as an array of filter coefficients.  Each row 
    % of the filter arrays contains the coefficients for four second order 
    % filters.  The transfer function for these four filters share the same
    % denominator (poles) but have different numerators (zeros).  All of these
    % coefficients are assembled into one vector that the ERBFilterBank 
    % can take apart to implement the filter.
    %
    % The filter bank contains "numChannels" channels that extend from
    % half the sampling rate (fs) to "lowFreq".  Alternatively, if the numChannels
    % input argument is a vector, then the values of this vector are taken to
    % be the center frequency of each desired filter.  (The lowFreq argument is
    % ignored in this case.)

    % Note this implementation fixes a problem in the original code by
    % computing four separate second order filters.  This avoids a big
    % problem with round off errors in cases of very small cfs (100Hz) and
    % large sample rates (44kHz).  The problem is caused by roundoff error
    % when a number of poles are combined, all very close to the unit
    % circle.  Small errors in the eigth order coefficient, are multiplied
    % when the eigth root is taken to give the pole location.  These small
    % errors lead to poles outside the unit circle and instability.  Thanks
    % to Julius Smith for leading me to the proper explanation.

    % Execute the following code to evaluate the frequency
    % response of a 10 channel filterbank.
    %	fcoefs = MakeERBFilters(16000,10,100);
    %	y = ERBFilterBank([1 zeros(1,511)], fcoefs);
    %	resp = 20*log10(abs(fft(y')));
    %	freqScale = (0:511)/512*16000;
    %	semilogx(freqScale(1:255),resp(1:255,:));
    %	axis([100 16000 -60 0])
    %	xlabel('Frequency (Hz)'); ylabel('Filter Response (dB)');

    % Rewritten by Malcolm Slaney@Interval.  June 11, 1998.
    % (c) 1998 Interval Research Corporation  
    """

    EarQ = 2 * Q10
    minBW = 24.7
    order = 1
    ERB = ((cf / EarQ) ** order + minBW ** order) ** (1 / order)
    B = 1.019 * 2 * np.pi * ERB

    T = 1 / fs

    A0 = T
    A2 = 0
    B0 = 1
    B1 = -2 * np.cos(2 * cf * np.pi * T) / np.exp(B * T)
    B2 = np.exp(-2 * B * T)

    A11 = -(2*T*np.cos(2*cf*np.pi*T)/np.exp(B*T) + 2*np.sqrt(3+2**1.5)*T*np.sin(2*cf*np.pi*T)/np.exp(B*T))/2
    A12 = -(2*T*np.cos(2*cf*np.pi*T)/np.exp(B*T) - 2*np.sqrt(3+2**1.5)*T*np.sin(2*cf*np.pi*T)/np.exp(B*T))/2
    A13 = -(2*T*np.cos(2*cf*np.pi*T)/np.exp(B*T) + 2*np.sqrt(3-2**1.5)*T*np.sin(2*cf*np.pi*T)/np.exp(B*T))/2
    A14 = -(2*T*np.cos(2*cf*np.pi*T)/np.exp(B*T) - 2*np.sqrt(3-2**1.5)*T*np.sin(2*cf*np.pi*T)/np.exp(B*T))/2


    tmp1 = -2 * np.exp(4j * cf * np.pi * T) * T + \
          2 * np.exp(-(B * T) + 2j * cf * np.pi * T) * T * \
          (np.cos(2 * cf * np.pi * T) - np.sqrt(3 - 2 ** 1.5) * \
            np.sin(2 * cf * np.pi * T))

    tmp2 = -2 * np.exp(4j * cf * np.pi * T) * T + \
          2 * np.exp(-(B * T) + 2j * cf * np.pi * T) * T * \
          (np.cos(2 * cf * np.pi * T) + np.sqrt(3 - 2 ** 1.5) * \
            np.sin(2 * cf * np.pi * T))

    tmp3 = -2 * np.exp(4j * cf * np.pi * T) * T + \
          2 * np.exp(-(B * T) + 2j * cf * np.pi * T) * T * \
          (np.cos(2 * cf * np.pi * T) - np.sqrt(3 + 2 ** 1.5) * \
            np.sin(2 * cf * np.pi * T))

    tmp4 = -2 * np.exp(4j * cf * np.pi * T) * T + \
          2 * np.exp(-(B * T) + 2j * cf * np.pi * T) * T * \
          (np.cos(2 * cf * np.pi * T) + np.sqrt(3 + 2 ** 1.5) * \
            np.sin(2 * cf * np.pi * T))

    gain = np.abs(tmp1 * tmp2 * tmp3 * tmp4 / \
                  (-2 / np.exp(2 * B * T) - 2 * np.exp(4j * cf * np.pi * T) + \
                  2 * (1 + np.exp(4j * cf * np.pi * T)) / np.exp(B * T)) ** 4)

    allfilts = np.ones(len(cf))
    fcoefs = np.column_stack((A0 * allfilts, A11, A12, A13, A14, A2 * allfilts, B0 * allfilts, B1, B2, gain))

    return fcoefs


def ERBFilterBank(x, fcoefs):
  
    """
    Original Matlab comments:
    % function output = ERBFilterBank(x, fcoefs)
    % Process an input waveform with a gammatone filter bank. This function 
    % takes a single sound vector, and returns an array of filter outputs, one 
    % channel per row.
    %
    % The fcoefs parameter, which completely specifies the Gammatone filterbank,
    % should be designed with the MakeERBFilters function.  If it is omitted,
    % the filter coefficients are computed for you assuming a 22050Hz sampling
    % rate and 64 filters regularly spaced on an ERB scale from fs/2 down to 100Hz.
    %

    % Malcolm Slaney @ Interval, June 11, 1998.
    % (c) 1998 Interval Research Corporation  
    % Thanks to Alain de Cheveigne' for his suggestions and improvements.
    """

    A0 = fcoefs[:,0]
    A11 = fcoefs[:,1]
    A12 = fcoefs[:,2]
    A13 = fcoefs[:,3]
    A14 = fcoefs[:,4]
    A2 = fcoefs[:,5]
    B0 = fcoefs[:,6]
    B1 = fcoefs[:,7]
    B2 = fcoefs[:,8]
    gain = fcoefs[:,9]

    output = np.zeros((gain.shape[0], x.shape[0]))

    for chan in range(gain.shape[0]):
        y1 = signal.lfilter([A0[chan]/gain[chan], A11[chan]/gain[chan], A2[chan]/gain[chan]],
                            [B0[chan], B1[chan], B2[chan]], x)
        y2 = signal.lfilter([A0[chan], A12[chan], A2[chan]],
                            [B0[chan], B1[chan], B2[chan]], y1)
        y3 = signal.lfilter([A0[chan], A13[chan], A2[chan]],
                            [B0[chan], B1[chan], B2[chan]], y2)
        y4 = signal.lfilter([A0[chan], A14[chan], A2[chan]],
                            [B0[chan], B1[chan], B2[chan]], y3)
        output[chan,:] = y4
    return output

def getFilterCoefs(CF,FS):

    # Number of frequency channels
    NF = len(CF)

    # Q10 according to barn owl data
    Q10 = 0.074 * np.power(CF, 0.5)

    # Get filter coefficients
    fcoefs = np.zeros((NF, 10))
    for n in range(NF):
        cf = np.array([CF[n], 20000])
        temp = MakeERBFilters(FS, cf, 100, Q10[n])
        fcoefs[n, :] = temp[0, :]
    return fcoefs


def low_pass_binaural(uL, uR, dt, tau):
    
    N = uL.shape[1]

    yL = np.zeros(uL.shape)
    yR = np.zeros(uR.shape)

    eps = dt/tau

    for n in range(N-1):
        yL[:,n+1] = yL[:,n] + eps*(-yL[:,n] + uL[:,n])
        yR[:,n+1] = yR[:,n] + eps*(-yR[:,n] + uR[:,n])

    return yL, yR

def low_pass(u, dt, tau):
    
    N = len(u)

    y = np.zeros(u.shape)

    eps = dt/tau

    for n in range(N-1):
        y[:,n+1] = y[:,n] + eps*(-y[:,n] + u[:,n])
        

    return y


def get_xcorr(ICcl, uL, uR, T, dt):
      
    t = np.arange(0, T+dt, dt)
    tt = np.tile(t, (len(ICcl.delays),1))
    t_delay = tt + ICcl.delays

    U_left = np.zeros((uL.shape[0], len(ICcl.delays), uL.shape[1]))
    U_right= np.zeros((uR.shape[0], len(ICcl.delays), uR.shape[1]))

    for n in range(uL.shape[0]):

        f_left = Akima1DInterpolator(t, uL[n,:])
        f_right = Akima1DInterpolator(t, uR[n,:])

        U_left[n,:] = f_left(t_delay)
        U_right[n,:] = np.flipud(f_right(t_delay))

    U = (U_left + U_right + ICcl.XCORR_BIAS)**2
    
    U[np.isnan(U)] = 0

    N = U.shape[2]

    x = np.zeros(U.shape)

    eps_x = dt/ICcl.XCORR_TAU

    for n in range(N-1):
        x[:,:,n+1] = x[:,:,n] + eps_x*(-x[:,:,n] + U[:,:,n]) 
   
    x = np.squeeze(x)
    
    return x


def rms(x):
    return np.sqrt(np.mean(x**2))



def get_iccl_inputs(ICcl, sL, sR, T, time_step):
    
    cochlea_left = ERBFilterBank(sL, ICcl.filter_coefs)
    cochlea_right = ERBFilterBank(sR, ICcl.filter_coefs)
  
    # Get the energy envelope in the filterbank outputs
    cochlea_energy_left, cochlea_energy_right = low_pass_binaural(cochlea_left**2, cochlea_right**2, time_step, ICcl.TAU_ENERGY)

    # ILD cue (z)
    y_left = np.zeros(cochlea_energy_left.shape)
    y_left[cochlea_energy_left > 1] = np.log10(cochlea_energy_left[cochlea_energy_left > 1])

    y_right = np.zeros(cochlea_energy_right.shape)
    y_right[cochlea_energy_right > 1] = np.log10(cochlea_energy_right[cochlea_energy_right > 1])

    z = 10 * (y_right - y_left)
    
    # Get the normalization term
    gain_left, gain_right = low_pass_binaural(cochlea_left**2, cochlea_right**2, time_step, ICcl.TAU_GAIN)
    
    #gain = 0.5 * (gain_left + gain_right)
    
    gain = 0.5 * (np.sqrt(gain_left) + np.sqrt(gain_right))
    
    # Normalize cochlea
    #scaled_cochlea_left = cochlea_left / np.sqrt(gain + ICcl.GAMMA)

    #scaled_cochlea_right = cochlea_right / np.sqrt(gain + ICcl.GAMMA)
    
    scaled_cochlea_left = cochlea_left / (gain + np.sqrt(ICcl.GAMMA))

    scaled_cochlea_right = cochlea_right / (gain + np.sqrt(ICcl.GAMMA))
    
    # Energy envelope in scaled signal 
    energy_left, energy_right = low_pass_binaural(scaled_cochlea_left**2, scaled_cochlea_right**2, time_step, ICcl.TAU_ENERGY)
    
    # Get cross correlation
    x = get_xcorr(ICcl, scaled_cochlea_left, scaled_cochlea_right, T, time_step)
    
    # Remove singleton dimensions
    energy_left = np.squeeze(energy_left)
    energy_right = np.squeeze(energy_right)
    z = np.squeeze(z)
    x = np.squeeze(x)
    
    return energy_left, energy_right, z, x


