from TTS.api import TTS
import simpleaudio as sa
import tempfile
import os

# Initialize the TTS model
tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=True, gpu=False)

# Get text input from the user
text_to_speak = input("Enter the text you want to convert to speech: ")

# Create a temporary file to save the speech
with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
    # Save the speech to the temporary file
    tts.tts_to_file(text_to_speak, file_path=tmp_file.name)

    # Play the audio file
    wave_obj = sa.WaveObject.from_wave_file(tmp_file.name)
    play_obj = wave_obj.play()
    play_obj.wait_done()  # Wait until sound has finished playing

# Clean up: remove the temporary file
os.remove(tmp_file.name)
