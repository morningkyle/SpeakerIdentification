import pyaudio
import wave


def recording_clip(clip_name, sec=15):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1, rate=44100, input=True, output=False)

    fout = wave.open(clip_name, 'w')
    fout.setnchannels(1)
    fout.setsampwidth(2)
    fout.setframerate(44100)

    CHUNK = 1024
    for i in range(0, int(44100 / CHUNK * sec)):
        data = stream.read(CHUNK)
        fout.writeframes(data)
    fout.close()

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("File {0} saved".format(clip_name))


def recording_train_clips(num_of_clips=3):
    for i in range(num_of_clips):
        label = input("input label{0}:".format(i))
        if len(label) <= 0:
            print("Recording stopped!")
            return
        print("Start recording for {0} seconds".format(15))
        recording_clip(label + '.wav')
    print("============ Done =============")


