import numpy as np
import matplotlib.pyplot as plt
from math import sin, pi
import scipy.signal as signal

input_signal = np.zeros(192)
for i in range(192):
    if 0 <= i < 64:
        input_signal[i] = sin(0.25 * pi * i)
    elif 128 <= i < 192:
        input_signal[i] = sin(0.5 * pi * i)

# input_signal = np.sin(0.25 * pi * n_first) + np.sin(0.5 * pi * n_second)
analysis_window = np.ones(16)  # u[n] - u[n - 16]

fig, ((sig), (pt_512), (dtdft_64), (dtdft_128),
      (dtdft_256)) = plt.subplots(5, 1)

sig.title.set_text(
    'Input signal: sin(0.25πn){u[n]-u[n-64]} + sin(0.5πn){u[n-128]-u[n-192]}')
sig.set_xlabel('Time [n]')
sig.plot(np.arange(len(input_signal)), input_signal)

# Generate the 512-pt DFT of the input signal
freq = np.linspace(-pi, pi, 512)
dft_512_pt = np.fft.fft(input_signal, 512)
pt_512.title.set_text('512-pt DFT of x[n]')
pt_512.set_xlabel('Frequency')
pt_512.plot(freq, abs(dft_512_pt))

# Generate the DTDFT at frequency 64 => TDFT[n, 2pi(64)/512)
freq_sampling_factor = 512
time_sampling_factor = 1
sampling_freq: float = lambda k: (2.0 * pi * k) / freq_sampling_factor

stft_64_f, stft_64_t, stft_64_Zxx = signal.stft(input_signal,
                                                sampling_freq(64),
                                                analysis_window,
                                                len(analysis_window))
dtdft_64.title.set_text('Discrete TDFT @ k = 64')
dtdft_64.set_xlabel('Time [n: sec]')
dtdft_64.set_ylabel('Frequency [Hz]')
dtdft_64.pcolormesh(stft_64_t, stft_64_f, np.abs(stft_64_Zxx), shading='auto')

stft_128_f, stft_128_t, stft_128_Zxx = signal.stft(input_signal,
                                                   sampling_freq(128),
                                                   analysis_window,
                                                   len(analysis_window))

dtdft_128.title.set_text('Discrete TDFT @ k = 128')
dtdft_128.set_xlabel('Time [n: sec]')
dtdft_128.set_ylabel('Frequency [Hz]')
dtdft_128.pcolormesh(stft_128_t,
                     stft_128_f,
                     np.abs(stft_128_Zxx),
                     shading='auto')

stft_256_f, stft_256_t, stft_256_Zxx = signal.stft(input_signal,
                                                   sampling_freq(256),
                                                   analysis_window,
                                                   len(analysis_window))

dtdft_256.title.set_text('Discrete TDFT @ k = 256')
dtdft_256.set_xlabel('Time [n: sec]')
dtdft_256.set_ylabel('Frequency [Hz]')
dtdft_256.pcolormesh(stft_256_t,
                     stft_256_f,
                     np.abs(stft_256_Zxx),
                     shading='auto')

plt.show()
