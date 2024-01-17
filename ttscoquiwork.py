from openai import OpenAI
from TTS.api import TTS
from pydub import AudioSegment
from pydub.playback import play

# Initialize the OpenAI client
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

# Initialize the TTS object
tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=True, gpu=True)

# Function to convert text to speech and play it
def speak(text):
    output_file = "response.wav"
    tts.tts_to_file(text, file_path=output_file)
    audio = AudioSegment.from_wav(output_file)
    play(audio)

history = [
    {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
    {"role": "user", "content": "Hello, introduce yourself to someone opening this program for the first time. Be concise."},
]

while True:
    completion = client.chat.completions.create(
        model="local-model",  # this field is currently unused
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

    print()
    user_input = input("> ")
    history.append({"role": "user", "content": user_input})
