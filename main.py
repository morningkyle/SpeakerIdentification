import scipy.io.wavfile as wav

# import recording
from model import IdentificationModel


model = IdentificationModel(labels=["ad", "news", "music"])
model.train()
print("Train finished.")


print("Start test:")
for label in model.labels:
    (rate, data) = wav.read(label + '.wav')
    start, end = int(len(data)/4), int(len(data)/2)
    prediction = model.test(data[start:end])
    print('{0} -> {1}'.format(label, prediction))


# for i in range(10):
#     recording.recording_clip('test.wav', sec=3)
#     (rate, data) = wav.read('test.wav')
#     label = model.test(data)
#     print("It is {0} speaking".format(label))

