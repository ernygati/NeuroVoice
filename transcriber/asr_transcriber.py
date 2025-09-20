import os
import shutil
import torch
import torchaudio
from torchaudio.functional import resample as tf_resample
from tqdm import tqdm
import pandas as pd

PUNCT_END = {".", "!", "?"}
WHISPER_SR = 16000


class ASRTranscriber:
    def __init__(self, asr_model, device="cuda"):
        self.asr_model = asr_model
        self.device = device
        
    @staticmethod
    def _to_sec(x, sr):  # accepts dict with "start","end"
        s, e = float(x["start"]), float(x["end"])
        return dict(start=s / sr, end=e / sr)

    @staticmethod
    def slice_audio_by_time(wav: torch.Tensor, sr: int, t0: float, t1: float) -> torch.Tensor:
        """Return slice wav[t0:t1] in seconds. Output is [T']."""
        start = max(int(sr * max(t0, 0.0)), 0)
        end = max(int(sr * max(t1, 0.0)), 0)
        return wav[start:end]

    @staticmethod
    def resample_to_16k(mono_1xT: torch.Tensor, src_sr: int) -> torch.Tensor:
        """Resample [1, T] to 16 kHz for Whisper."""
        return tf_resample(mono_1xT, orig_freq=src_sr, new_freq=WHISPER_SR)

    @staticmethod
    def split_words_into_sentences(words, max_duration: float, buffer_sec_start, buffer_sec_end):
        """
        words: list of {"word","start","end"} (seconds, chunk-local).
        End sentence on . ! ? or when duration exceeds max_duration.
        """
        sentences = []
        cur = []
        for w in words:
            token = (w.get("word") or "").strip()
            if not token:
                continue
            start = float(w.get("start", 0.0))
            end = float(w.get("end", 0.0))
            cur.append({"word": token, "start": start, "end": end})

            duration = cur[-1]["end"] - cur[0]["start"]
            last_tok = cur[-1]["word"]
            end_now = (last_tok[-1] in PUNCT_END)

            if end_now:
                s0 = max(cur[0]["start"] - buffer_sec_start, 0.0)
                s1 = cur[-1]["end"] + buffer_sec_end
                text = " ".join(x["word"] for x in cur).strip()
                if text:
                    sentences.append({"start": s0, "end": s1, "text": text})
                cur = []

        if cur:
            s0 = max(cur[0]["start"] - buffer_sec_start, 0.0)
            s1 = cur[-1]["end"] + buffer_sec_end
            text = " ".join(x["word"] for x in cur).strip()
            if text:
                sentences.append({"start": s0, "end": s1, "text": text})

        return sentences

    @staticmethod
    def to_mono_1xT(wav: torch.Tensor) -> torch.Tensor:
        """Ensure mono float tensor shaped [1, T]."""
        if wav.dim() == 2:  # [C, T]
            wav = wav.mean(dim=0)
        return wav.unsqueeze(0) if wav.dim() == 1 else wav

    @staticmethod
    def save_wav(out_dir, base_name, idx, sr, audio, text, speaker, bad_wav=False):
        out_dir = os.path.join(out_dir, "wavs") if not bad_wav else os.path.join(out_dir, "badwavs")
        os.makedirs(out_dir, exist_ok=True)
        if idx == 0 and os.listdir(out_dir):
            shutil.rmtree(out_dir)
        fname = f"{base_name}_{str(idx).zfill(4)}.wav"
        abs_path = os.path.join(out_dir, fname)
        audio_1xT = ASRTranscriber.to_mono_1xT(audio).cpu()
        torchaudio.save(abs_path, audio_1xT, sr)
        if not bad_wav:
            return {
                "audio_file": f"wavs/{fname}",  # keep the "wavs/" convention
                "text": text,
                "speaker_name": speaker,
            }
        return None

    @staticmethod
    def metadata_split_and_save(metadata, output_folder, train_split):
        # Split into train/eval
        train_size = max(1, int(len(metadata) * train_split))
        train_df = metadata.sample(train_size, random_state=42)
        eval_df = metadata.drop(train_df.index)
        
        # Save metadata files
        train_path = os.path.join(output_folder, "metadata_train.csv")
        eval_path = os.path.join(output_folder, "metadata_eval.csv")
        train_df.to_csv(train_path, sep="|", index=False)
        eval_df.to_csv(eval_path, sep="|", index=False)
        print(f"All wavs and metadata are saved in {output_folder} !")
        return train_path, eval_path

    @torch.inference_mode()
    def transcribe_and_save_vad_segments(
        self,
        wav: torch.Tensor,
        sr: int,
        vad_segments,                    # [{"start": sample_idx|sec, "end": sample_idx|sec}, ...]
        base_name: str,
        out_dir: str,                    # folder where wavs/* will be written
        speaker: str = "Divertito",
        min_duration: float = 2.0,
        max_duration: float = 10.0,
        buffer_sec_start: float = 0.15,  # small padding at the beginning of each wav
        buffer_sec_end: float = 0.10,
        language: str | None = "ru",
        task: str = "transcribe",
        use_amp: bool = True,
        fp16: bool = True,
        collect_bpe_words: bool = False,
        train_split: float = 0.85
    ):
        """
        End-to-end (no temp files):
          1) slice VAD chunk from original wav (any sr)
          2) resample to 16k and pass tensor directly to Whisper
          3) split words into sentence-sized items
          4) save sentence slices (original sr) + text, build metadata
        Returns:
          metadata: dict of lists
          whisper_words: flat token list if collect_bpe_words=True
        """
        os.makedirs(out_dir, exist_ok=True)

        metadata = {"audio_file": [], "text": [], "speaker_name": []}
        whisper_words = []
        counter = 0
        amp_ctx = torch.cuda.amp.autocast if (use_amp and self.device == "cuda") else torch.cpu.amp.autocast
        
        with amp_ctx(enabled=use_amp):
            for vi, raw_seg in enumerate(tqdm(vad_segments)):
                seg = self._to_sec(raw_seg, sr)
                # 1) slice VAD chunk in original sr
                chunk_orig = self.slice_audio_by_time(wav.squeeze(0), sr, seg["start"], seg["end"])
                if chunk_orig.size(-1) < sr * min_duration:  # <2s
                    continue

                # 2) resample to 16k for Whisper
                chunk_16k = self.resample_to_16k(chunk_orig, sr)

                # 3) transcribe tensor directly
                result = self.asr_model.transcribe(
                    chunk_16k,
                    language=language,
                    task=task,
                    word_timestamps=True,
                    fp16=fp16
                )
                if not result or not result.get("segments"):
                    continue

                # gather words (chunk-local seconds)
                words = []
                for segm in result["segments"]:
                    for w in segm.get("words", []):
                        token = (w.get("word") or "").strip()
                        if not token:
                            continue
                        start = float(w.get("start", 0.0))
                        end = float(w.get("end", 0.0))
                        words.append({"word": token, "start": start, "end": end})
                        if collect_bpe_words:
                            whisper_words.append(token)

                if not words:
                    continue

                # 4) sentence split
                sentences = self.split_words_into_sentences(
                    words, max_duration,
                    buffer_sec_start, buffer_sec_end
                )

                # 5) save each sentence slice from *original-sr* chunk
                for s in sentences:
                    s0, s1, text = s["start"], s["end"], s["text"]
                    sent_audio = self.slice_audio_by_time(chunk_orig.squeeze(0), sr, s0, s1)
                    if sent_audio.size(-1) < sr * min_duration or sent_audio.size(-1) > sr * max_duration:  # filter out <2s and > 10s 
                        row = self.save_wav(out_dir, base_name, counter, sr, sent_audio, text, speaker, bad_wav=True)
                    else:
                        row = self.save_wav(out_dir, base_name, counter, sr, sent_audio, text, speaker)
                    if row is not None:
                        for k, v in row.items():
                            metadata[k].append(v)
                    counter += 1

        metadata_df = pd.DataFrame(metadata)
        train_metadata_path, eval_metadata_path = self.metadata_split_and_save(metadata_df, out_dir, train_split)
        if collect_bpe_words:
            return train_metadata_path, eval_metadata_path, whisper_words
        else:
            return train_metadata_path, eval_metadata_path


if __name__ == "__main__":
    # Example usage
    import whisper
    
    # Initialize Whisper model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("medium").to(device)
    
    # Create transcriber instance
    transcriber = ASRTranscriber(model, device=device)
    
    # Load sample audio
    audio_path = "sample.wav"
    wav, sr = torchaudio.load(audio_path)
    wav = transcriber.to_mono_1xT(wav).to(device)
    
    # Example VAD segments (in seconds)
    vad_segments = [
        {"start": 0.0, "end": 5.0},
        {"start": 5.0, "end": 10.0},
    ]
    
    # Transcribe and save segments
    train_metadata_path, eval_metadata_path = transcriber.transcribe_and_save_vad_segments(
        wav=wav,
        sr=sr,
        vad_segments=vad_segments,
        base_name="sample_audio",
        out_dir="output",
        speaker="example_speaker",
        min_duration=2.0,
        max_duration=10.0,
        language="en",
        train_split=0.8
    )
    
    print("Transcription complete. Metadata:")
    print(metadata)