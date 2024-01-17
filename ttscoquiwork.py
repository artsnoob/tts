from TTS.api import TTS
from pydub import AudioSegment
from pydub.playback import play

# Create a TTS object with the specified model
tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=True, gpu=True)

# Prompt the user for input
user_input = input("Enter the text you want to convert to speech: ")

# Define the output file path
output_file = "OUTPUT.wav"

# Use the TTS object to convert the input text to speech and save it to a file
tts.tts_to_file(user_input, file_path=output_file)

# Load the audio file
audio = AudioSegment.from_wav(output_file)

# Play the audio
play(audio)
