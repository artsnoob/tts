from TTS.api import TTS

# Create a TTS object with the specified model
tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=True, gpu=True)

# Prompt the user for input
user_input = input("Enter the text you want to convert to speech: ")

# Use the TTS object to convert the input text to speech and save it to a file
tts.tts_to_file(user_input, file_path="OUTPUT.wav")
