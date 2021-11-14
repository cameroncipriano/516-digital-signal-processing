import numpy as np
import matplotlib.pyplot as plt
from math import pi

n = np.arange(64)  # u[n] - u[n-64]
input_signal = np.sin(0.125 * n * pi + 0.1 * pi)

fig, ((signal), (pt_64), (pt_128), pt_512) = plt.subplots(4, 1)
signal.title.set_text('Input signal: sin(0.125πn + 0.1π)')
signal.set_xlabel('Time [n]')
signal.set_ylabel('Value')
signal.plot(n, input_signal)

freq = np.linspace(-pi, pi, 64)
dft_64_pt = np.fft.fft(input_signal, 64)
pt_64.title.set_text('64-Pt DFT of x[n]')
pt_64.set_xlabel('Frequency')
pt_64.plot(freq, abs(dft_64_pt))

freq = np.linspace(-pi, pi, 128)
dft_128_pt = np.fft.fft(input_signal, 128)
pt_128.title.set_text('128-Pt DFT of x[n]')
pt_128.set_xlabel('Frequency')
pt_128.plot(freq, abs(dft_128_pt))

freq = np.linspace(-pi, pi, 512)
dft_512_pt = np.fft.fft(input_signal, 512)
pt_512.title.set_text('512-Pt DFT of x[n]')
pt_512.set_xlabel('Frequency')
pt_512.plot(freq, abs(dft_512_pt))

plt.show()