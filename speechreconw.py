'''import sounddevice as sd
import numpy as np
import time
import whisper
import scipy.io.wavfile
import io

SAMPLE_RATE = 16000
DURATION = 5  # seconds
NUM_MFCC = 40
model = whisper.load_model("base") 
try:
    while True:
        print("\n Listening for 5 seconds...")
        audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()
        result = model.transcribe(audio)
        text=result["text"].strip()
        
        if "help"in text.lower():
            print("sosdtected")
        else:
            print(text)
except KeyboardInterrupt:
    print("\n Stopped by user.")'''
import whisper
import numpy as np

# Load the Whisper model once
whisper_model = whisper.load_model("base")

def detect_sos_with_whisper(audio_array: np.ndarray, sample_rate: int = 16000) -> bool:
    """
    Detects if the word 'help' is present in the given audio using OpenAI's Whisper.

    Parameters:
        audio_array (np.ndarray): The audio waveform (mono) as a NumPy array.
        sample_rate (int): Sample rate of the audio.

    Returns:
        bool: True if 'help' is detected, False otherwise.
    """
    # Ensure float32 and shape [length]
    if audio_array.ndim > 1:
        audio_array = audio_array.flatten()
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)

    # Transcribe using Whisper
    result = whisper_model.transcribe(audio_array, fp16=False, language="en", task="transcribe")
    text = result["text"].strip()
    print(f"Transcribed Text: {text}")

    if "help" in text.lower():
        print("🆘 SOS detected!")
        return True
    else:
        print("✅ No SOS detected.")
        return False
