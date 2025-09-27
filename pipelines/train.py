import os
import yaml
import torch
import torchaudio
from pathlib import Path
import whisper
from models.VAD.vad_processer import VADProcessor
from transcriber.asr_transcriber import ASRTranscriber
from models.tts.xtts_finetuner import XTTSFinetuner, OUTPUT_SAMPLE_RATE


class TrainPipeline:
    def __init__(self,config):
        self.config=config
        self.speaker = config["name"]
        self.wav_filename = os.path.basename(config["data"]["base_wav_path"]).split(".")[0]

        self.train_data_dir = os.path.join(config["data"]["main_dir"],self.wav_filename)
        self.MIN_DURATION = config["audio"]["min_duration"]
        self.MAX_DURATION = config["audio"]["max_duration"]
        self.device = torch.device(config["device"])
        self.finetuner = XTTSFinetuner(save_checkpoint_path=self.config["data"]["weights_folder"])

    def _check_train_files(self, train_data_dir):
        mandatory_files = ["metadata_train.csv", "metadata_eval.csv", "wavs"]
        flag = True
        if all(mf in os.listdir(train_data_dir) for mf in mandatory_files):
            print("All mandatory files are ready!")
        else:
            flag = False
        if os.listdir(f"{train_data_dir}/wavs"):
            print("wavs/ folder is not empty!")
        else:
            flag = False
        return os.path.join(train_data_dir,mandatory_files[0]),\
            os.path.join(train_data_dir,mandatory_files[1])

    #готовим данные для обучения если их нет
    def _preprocess_data(self,train_data_dir):
        if not self._check_train_files(train_data_dir):
            vad_processer = VADProcessor(self.device,
                                min_duration=self.MIN_DURATION, 
                                max_duration=self.MAX_DURATION)
            # Load and prepare audio
            audio_path=self.config["base_wav_path"]
            wav, sr = torchaudio.load(audio_path)
            wav = wav.to(self.device)
            if wav.size(0) != 1:  # Convert stereo to mono
                wav = torch.mean(wav, dim=0, keepdim=True)
            wav = wav.squeeze()
            #готовим модель для транскрипции
            asr_model = whisper.load_model(self.config["models"]["whisper"], 
                                        device=self.device,
                                        download_root="models/whisper")
            #готовим разбиение wav на VAD-сегменты
            vad_segments = vad_processer.get_vad_segments(wav,sr)

            # готовим wav-файлы для обучения модели и их транскрипты
            transcriber = ASRTranscriber(asr_model, device=self.device)
            train_metadata_path, eval_metadata_path = transcriber.transcribe_and_save_vad_segments(
                    wav=wav,
                    sr=sr,
                    vad_segments=vad_segments,
                    base_name=self.wav_filename,
                    out_dir=train_data_dir,
                    speaker=speaker,
                    min_duration=self.MIN_DURATION,
                    max_duration=self.MAX_DURATION,
                    language="ru",
                    train_split=0.85
                )
        else:
            train_metadata_path, eval_metadata_path = self._check_train_files(train_data_dir)
        return train_metadata_path, eval_metadata_path

    def train(self):
        train_metadata_path, eval_metadata_path =self._preprocess_data(self.train_data_dir)
        os.makedirs(self.config["data"]["weights_folder"], exist_ok=True)
        
        base_model_path =  self.finetuner.load_base_model()
        checkpoint_path =  self.finetuner.finetune_model(
            self.device,
            base_model_path,
            train_metadata=Path(train_metadata_path) , 
            eval_metadata = Path(eval_metadata_path),
            num_epochs=self.config["train"]["epochs"],
            batch_size=self.config["train"]["batch_size"],
            checkpoint_pth_path=self.config["data"]["checkpoint_path"],
        )