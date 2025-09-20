import os
from datetime import datetime
import glob
from IPython.display import display, Audio
import soundfile as sf
import librosa
from pydub import AudioSegment
import tempfile
from NeuroVoice.utils.quantizer import Quantizer
from NeuroVoice.models.tts.xtts_inference import XTTSInference, OUTPUT_SAMPLE_RATE

TELEGRAM_SAMPLE_RATE = 48000

class VoiceGenerator:
    def __init__(self):
        self.XTTSInf = XTTSInference()
        self.ref_wav_list = glob.glob("DivertitoVoice/data/*.wav")
        
    def make_model(self):
        xinference = XTTSInference()
        print("Loading voice model...")
        vmodel = xinference.load_model( config_path= "./NeuroVoice/models/tts/base/coqui/XTTS-v2/config.json",
                        checkpoint_path="./NeuroVoice/models/tts/finetuned/xtts/Divertito/best_model_v2.pth",
                        vocab_path="./NeuroVoice/models/tts/base/coqui/XTTS-v2/vocab.json",
                        speaker_xtts_path="./NeuroVoice/models/tts/base/coqui/XTTS-v2/speakers_xtts.pth"
                
                    )
        quantizer = Quantizer(vmodel)
        vmodel = quantizer.accelerate()
        print("Voice model loaded successfully!")
        return vmodel
    
    def voice_text(self, vmodel, text, save_path, ext="ogg"):
        _, wav = self.XTTSInf.xtts_generate_long(vmodel, text, self.ref_wav_list)
        now = datetime.now()
        data = now.strftime("%Y-%m-%d")
        savefilename=os.path.join(save_path, f"{data}_out.{ext}")
        if ext == "wav":
            sf.write(savefilename, wav, OUTPUT_SAMPLE_RATE)
        elif ext=='ogg':
            wav = librosa.resample(wav, orig_sr=OUTPUT_SAMPLE_RATE, target_sr=TELEGRAM_SAMPLE_RATE)
            self._save_ogg(savefilename, wav, TELEGRAM_SAMPLE_RATE)
        return savefilename

        
    def _save_ogg(self,save_path, audio_array, sample_rate):
        """
        Save numpy array as OGG by first saving as WAV then converting for TG bot
        """
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_filename = temp_wav.name
        
        try:
            # if len(audio_array.shape) == 1:
            #      audio_array = audio_array[None,:]
            sf.write(temp_filename, audio_array, sample_rate)
            
            # Convert WAV to OGG using pydub
            audio = AudioSegment.from_wav(temp_filename)
            audio.export(save_path, format="ogg",  codec="libopus")
            print(f"Audio saved as OGG: {save_path}")
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

        
    
