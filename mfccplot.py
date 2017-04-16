import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


y, sr = librosa.load("../Data/train/train2.wav")

mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
D = np.abs(librosa.stft(y))**2
S = librosa.feature.melspectrogram(S=D)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
#plt.show()

'''
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S, ref=np.max),y_axis='mel', fmax=8000,x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')'''
plt.savefig('elephant_mfcc.png')
