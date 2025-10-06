import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.signal import stft
import pywt

fs = 1000
N = 3000

def generate_sinusoidal_signal(frequency=4, duration=3, sampling_rate=1000):
    """
    Generate a sinusoidal signal.

    Args:
        frequency (float): Frequency of the sine wave in Hz.
        duration (float): Duration of the signal in seconds.
        sampling_rate (int): Number of samples per second.

    Returns:
        t (np.ndarray): Time vector.
        signal (np.ndarray): Sinusoidal signal.
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * frequency * t)
    return t, signal

def generate_windowed_sinusoidal(frequency=12, duration=3, sampling_rate=1000, start=1, end=2):
    """
    Generate a sinusoidal signal with given frequency between 'start' and 'end' seconds, zero elsewhere.

    Args:
        frequency (float): Frequency of the sine wave in Hz.
        duration (float): Total duration of the signal in seconds.
        sampling_rate (int): Number of samples per second.
        start (float): Start time of the sinusoidal part in seconds.
        end (float): End time of the sinusoidal part in seconds.

    Returns:
        t (np.ndarray): Time vector.
        signal (np.ndarray): Windowed sinusoidal signal.
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = np.zeros_like(t)
    mask = (t >= start) & (t < end)
    signal[mask] = np.sin(2 * np.pi * frequency * t[mask])
    return t, signal

def generate_chirp_signal(f0=30, f1=0, duration=3, sampling_rate=1000):
    """
    Generate a linear chirp signal.

    Args:
        f0 (float): Starting frequency of the chirp in Hz.
        f1 (float): Ending frequency of the chirp in Hz.
        duration (float): Duration of the signal in seconds.
        sampling_rate (int): Number of samples per second.

    Returns:
        t (np.ndarray): Time vector.
        signal (np.ndarray): Chirp signal.
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    k = (f1 - f0) / duration  # Chirp rate
    signal = np.sin(2 * np.pi * (f0 * t + 0.5 * k * t**2))
    return signal

# Example usage and plot
#if __name__ == "__main__":
t, signal1 = generate_sinusoidal_signal(frequency=4, duration=3, sampling_rate=1000)
t, signal2 = generate_sinusoidal_signal(18, 3, 1000)
t, signal3 = generate_windowed_sinusoidal(12, 3, 1000, 1, 2)
t, signal4 = generate_windowed_sinusoidal(15, 3, 1000, 1, 2.)

chirp = generate_chirp_signal(1, 50, 3, 1000)
gaussian_noise = np.random.normal(0, 1, len(t))

mother_signal = signal1 + signal2 + signal3 + signal4 + 2 + gaussian_noise

plt.plot(t, chirp)
plt.plot(t, mother_signal)



# Compute FFT of the mother signal
fft_vals = sp.fft.fft(mother_signal)/len(mother_signal)
fft_freqs = sp.fft.fftfreq(len(mother_signal), 1/1000)

# Prepare figure with 3 subplots
#fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# FFT plot (limit frequency axis for readability)
plt.figure(figsize=(10, 4))
plt.plot(fft_freqs[:len(fft_freqs)//2], np.abs(fft_vals[:len(fft_vals)//2]))
plt.xlim(0, 50)
plt.title("FFT of Mother Signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.tight_layout()
#plt.show()

# # STFT plot (limit frequency axis for readability)
# f, t_stft, Zxx = stft(mother_signal, fs=1000, nperseg= 1024)
# plt.figure(figsize=(10, 4))
# pcm = plt.pcolormesh(t_stft, f, np.abs(Zxx), shading='gouraud')
# plt.ylim(0, 50)
# plt.title('STFT Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.colorbar(pcm, label='Magnitude')
# plt.tight_layout()
# #plt.show()


# freqs = np.linspace(1, 50, 200)  # Frequency range of interest
# wavelet = 'cmor100.0-1.0'  # Complex Morlet wavelet
# # Wavelet Transform plot (limit frequency axis for readability)
# scales = pywt.central_frequency(wavelet) * fs / freqs
# # Use complex Morlet wavelet
# coefficients, frequencies = pywt.cwt(mother_signal, scales, wavelet, sampling_period=1/fs)

# # === Plot the CWT scalogram ===
# plt.figure(figsize=(10,6))
# plt.imshow(np.abs(coefficients), extent=[0, N/fs, frequencies[-1], frequencies[0]],
#            aspect='auto', cmap='jet')
# plt.colorbar(label='Magnitude')
# plt.title("Wavelet Transform (CWT - Morlet)")
# plt.xlabel("Time [s]")
# plt.ylabel("Frequency [Hz]")
# plt.ylim(0, 50)  # focus on the frequency band of interest
# plt.tight_layout()


# --- Wigner-Ville Distribution ---
from tftb.processing import WignerVilleDistribution

wvd = WignerVilleDistribution(mother_signal)
tfr, t_wvd, f_wvd = wvd.run()
plt.figure(figsize=(10, 6))
plt.imshow(np.abs(tfr), extent=[t_wvd[0], t_wvd[-1], f_wvd[0]*fs, f_wvd[-1]*fs], aspect='auto', origin='lower', cmap='jet')
plt.title("Wigner-Ville Distribution")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.ylim(0, 50)
plt.colorbar(label='Magnitude')
plt.tight_layout()
#plt.show()

# --- Hilbert Transform (analytic signal & envelope) ---
analytic_signal = sp.signal.hilbert(mother_signal)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = np.diff(instantaneous_phase) * fs / (2.0*np.pi)

plt.figure(figsize=(10, 4))
plt.plot(t, mother_signal, label='Mother Signal')
plt.plot(t, amplitude_envelope, label='Envelope')
plt.title("Hilbert Transform: Envelope of Mother Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
#plt.show()

plt.figure(figsize=(10, 4))
plt.plot(t[1:], instantaneous_frequency)
plt.title("Hilbert Transform: Instantaneous Frequency")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.ylim(0, 50)
plt.tight_layout()
#plt.show()


plt.show()



