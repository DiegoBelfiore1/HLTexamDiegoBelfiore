from pyannote.audio import Pipeline
from pydub import AudioSegment
import speech_recognition as sr
recognizer = sr.Recognizer()

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="r")
print ("Call Model OK")

c = 0

# apply model to divide the audios and get text
while c < 10:
    listSPEAKER_00 = []
    listSPEAKER_01 = []
    diarization = pipeline(f"audio{c}.mp3")
    audio = AudioSegment.from_file(f"audio{c}.mp3")
    print ("analysing audio number", c)
    # print to be sure which audio is being analysed
    for turn, _, speaker in diarization.itertracks(yield_label=True):

        start_time = turn.start * 1000  
        end_time = turn.end * 1000   
        audioportion = audio[start_time:end_time]
        audioportion.export("temp_audio.wav", format="wav")

        with sr.AudioFile("temp_audio.wav") as source:
            recognizer.pause_threshold = 10
            audio2 = recognizer.listen(source, timeout=2400)
            text = recognizer.recognize_whisper(audio2)
            if speaker == 'SPEAKER_00':
                listSPEAKER_00.append(text)
                print ("I'm writing speaker 0")
            elif speaker == 'SPEAKER_01':
                listSPEAKER_01.append(text)
                print ("I'm writing speaker 1")

    with open(f"speaker{c}0.txt", "w") as file:
        file.write('\n'.join(listSPEAKER_00))

    with open(f"speaker{c}1.txt", "w") as file:
        file.write('\n'.join(listSPEAKER_01))
    c = c + 1

else: print ("work is done")