3 versions right now; 

- CoquiWork: Working on Windows and Mac (tested on Apple Silicon). Text input, TTS output from LLM. Using Coqui Text to Speech.
- WhisperWindows: STT -> LLM -> TTS for Windows. Using Whisper STT and Coqui Text to Speech.
- WisperMac: STT -> LLM -> TTS for Mac (tested on Apple Silicon). Using Whisper STT and Coqui Text to Speech.

******************************************************

*** Setup a LLM server within LM Studio and choose the LLM model that you want to use. ***

To use on Mac Silicon, set "GPU=True" to False.

To use on Windows, install Nvidia Cuda Toolkit.

For both platforms; create a venv and install the requirements.txt.

Also, install Pytorch; https://pytorch.org/get-started/locally/

******************************************************

Future updates:
- Add Speech to Text for full immersion.

******************************************************

Current version:

Whisperwindows.py V0.4 (18/01/2024)
Working on Windows with GPU enabled, quicker then on my Mac, working with attached headset.

******************************************************

Previous versions:

Whispermac.py V0.3 (18/01/2024)
Added Whisper input so you can actually talk to the model. Using "Whisper-Large-V3" because it seems to be the most reliable, but slow.
Turn on GPU acceleration when on PC.

Coqui Work V0.3 (17/01/2024)
Added TTS to the feedback provided by the LLM.

V0.2 (17/01/2024)
On Windows, get immediate feedback of the imported text.

V0.1 (17/01/2024)
Just the working TTS script that saves the text as a .WAV

