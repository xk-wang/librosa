import librosa
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

audiopath = '../samples/p225_001.wav'
audio, sr = librosa.load(audiopath, sr=32000, mono=True)
print(sr)

start = time.time()
# window='hann', pad_mode='reflect',
mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512,  
        power=2.0, n_mels=229, fmin=30, fmax=8000, htk=True)
end = time.time()
print('the mel cal time: ', (end-start)*1000, ' ms')
print(mels.shape)
res = mels.sum(axis=1)
for i in range(10):
    print('%.6f'%res[i], end='\t')
print()

cqts = librosa.cqt(audio, sr=sr, hop_length=512, fmin=30, n_bins=256, bins_per_octave=48);
cqts = np.abs(cqts)
print(cqts.shape)
res = cqts.sum(axis=1)
for i in range(10):
    print('%.6f'%res[i], end='\t')
print()