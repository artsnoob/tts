import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from openai import OpenAI
from TTS.api import TTS
from pydub import AudioSegment
from pydub.playback import play
import sounddevice as sd
import numpy as np
import scipy.io.wavfile
from scipy.io.wavfile import write
import io

# Initialize the OpenAI client
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

# Initialize the TTS object
tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=True, gpu=True)

# Initialize the Whisper model for Speech-to-Text
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

# Function to convert text to speech and play it
def speak(text):
    if not text.strip():
        print("No text provided for speech synthesis.")
        return

    output_file = "response.wav"
    tts.tts_to_file(text, file_path=output_file)
    audio = AudioSegment.from_wav(output_file)
    play(audio)

# Function to record audio
def record_audio(duration=5, sample_rate=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    print("Recording finished")
    return audio

# Function to ensure the audio is mono
def ensure_mono(audio, sample_rate=16000):
    if audio.shape[1] != 1:
        audio = np.mean(audio, axis=1, keepdims=True)
    print("Audio shape after mono conversion:", audio.shape)
    return audio

# Function to convert the recorded audio to WAV format
def convert_to_wav(audio, sample_rate=16000):
    audio = (np.iinfo(np.int16).max * (audio/np.abs(audio).max())).astype(np.int16)
    wav_io = io.BytesIO()
    write(wav_io, sample_rate, audio)
    wav_io.seek(0)
    return wav_io

# Function to convert speech to text
def listen():
    audio = record_audio()
    mono_audio = ensure_mono(audio)
    wav_io = convert_to_wav(mono_audio)
    wav_io.seek(0)
    sample_rate, audio_data = scipy.io.wavfile.read(wav_io)
    print("Shape of audio data:", audio_data.shape)
    print("Data type:", audio_data.dtype)
    result = pipe({"array": audio_data, "sampling_rate": sample_rate})
    return result["text"]

history = [
    {"role": "system", "content": "You are an intelligent human. You are to act as a human, not as an AI Language Model."},
    {"role": "user", "content": "Hello, introduce yourself to someone opening this program for the first time. Act human."},
]

while True:
    print("Speak your input:")
    user_input = listen()
    print(f"You said: {user_input}")

    if user_input.strip():
        history.append({"role": "user", "content": user_input})

        # Filter out empty messages from history
        history = [msg for msg in history if msg["content"].strip()]

        # Check if history has valid content
        if history:
            completion = client.chat.completions.create(
                model="local-model",
                messages=history,
                temperature=0.7,
                stream=True,
            )

            new_message = {"role": "assistant", "content": ""}

            for chunk in completion:
                if chunk.choices[0].delta.content:
                    response_text = chunk.choices[0].delta.content
                    print(response_text, end="", flush=True)
                    new_message["content"] += response_text

            history.append(new_message)

            # Speak the response
            speak(new_message["content"])
        else:
            print("No valid history to process.")
