from scipy.io.wavfile import read
import pandas as pd

a = pd.Series(read("test.wav")[1])