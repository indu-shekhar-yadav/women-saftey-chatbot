import deepspeech
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import os

class SpeechToText:
    def __init__(self, model_path='../deepspeech-0.9.3-models.pbmm', scorer_path='../deepspeech-0.9.3-models.scorer'):
        # Resolve the absolute path to the model files
        base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of speech_to_text.py (src/)
        model_path = os.path.join(base_dir, model_path)
        scorer_path = os.path.join(base_dir, scorer_path)
        
        print(f"Resolved model path: {model_path}")
        print(f"Resolved scorer path: {scorer_path}")
        
        # Check if files exist
        if not os.path.exists(model_path):
            print(f"Model file does not exist at: {model_path}")
        if not os.path.exists(scorer_path):
            print(f"Scorer file does not exist at: {scorer_path}")
        
        if not os.path.exists(model_path) or not os.path.exists(scorer_path):
            raise FileNotFoundError("DeepSpeech model or scorer file not found. Please download them.")
        
        print("Loading DeepSpeech model...")
        self.model = deepspeech.Model(model_path)
        print("DeepSpeech model loaded.")
        
        print("Enabling external scorer...")
        self.model.enableExternalScorer(scorer_path)
        print("External scorer enabled.")

    def record_audio(self, duration=5, fs=16000):
        print("Recording... Speak now!")
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        return audio.flatten()

    def transcribe(self, audio):
        return self.model.stt(audio)

# Example usage
if __name__ == "__main__":
    try:
        stt = SpeechToText()
        audio = stt.record_audio()
        text = stt.transcribe(audio)
        print(f"Transcribed text: {text}")
    except Exception as e:
        print(f"Error in main: {str(e)}")