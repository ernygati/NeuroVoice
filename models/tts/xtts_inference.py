import os
import torch
from tqdm import tqdm
import numpy as np
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig
from ...utils.long_text_splitter import LongTextSplitter
from typing import List, Tuple

OUTPUT_SAMPLE_RATE = 24000

class XTTSInference:        
    def __init__(self):
        self.longtTS = LongTextSplitter()
    
    def load_model(self, config_path, checkpoint_path, vocab_path,speaker_xtts_path):
        """Load a trained XTTS model for inference."""
        config = XttsConfig()
        config.load_json(config_path)
        
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_path=checkpoint_path,
            vocab_path=vocab_path,
            use_deepspeed=False,
            speaker_file_path= speaker_xtts_path
        )
        
        return model.to("cpu")
    
    @torch.inference_mode()
    def generate_audio(self, model, text, reference_audio, language="ru"):
        """Generate audio from text using the finetuned model."""
        # Get conditioning latents from reference audio
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[reference_audio],
            gpt_cond_len=model.config.gpt_cond_len,
            max_ref_length=model.config.max_ref_len
        )
        
        # Generate audio
        out = model.inference(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding
        )
        
        return out["wav"]
        

    def _crossfade_concat(self, wavs: List[np.ndarray], sr: int, fade_ms: int=25) -> np.ndarray:
        """Simple linear crossfade to avoid clicks between chunks."""
        if not wavs: return np.zeros(0, dtype=np.float32)
        out = wavs[0].astype(np.float32)
        fade = max(1, int(sr * fade_ms / 1000))
        for w in wavs[1:]:
            w = w.astype(np.float32)
            if len(out) < fade or len(w) < fade:
                out = np.concatenate([out, w], axis=0); continue
            # apply crossfade
            a = out[-fade:].copy()
            b = w[:fade].copy()
            ramp = np.linspace(1.0, 0.0, fade, dtype=np.float32)
            out[-fade:] = a * ramp
            out = np.concatenate([out, b * (1.0 - ramp), w[fade:]], axis=0)
        # clamp
        np.clip(out, -1.0, 1.0, out=out)
        return out

    # ---------- main generation ----------
    @torch.inference_mode()
    def xtts_generate_long(self, 
                           model, 
                           text: str,
                           reference_audio: str, 
                           language: str="ru",
                            pause_ms: int=120,
                            temperature=0.3,# small pause between chunks (before crossfade)
                            **hf_generate_kwargs) -> Tuple[List[str], np.ndarray]:
        """
        Splits text safely and generates chunk-by-chunk with shared conditioning.
        Returns (chunks, wav).
        """
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[reference_audio] if isinstance(reference_audio, str) else reference_audio,
            gpt_cond_len=model.config.gpt_cond_len,
            max_ref_length=model.config.max_ref_len
        )

        chunks = self.longtTS.chunk_text_for_xtts(text, language)

        sr = getattr(model, "output_sample_rate", OUTPUT_SAMPLE_RATE)  # adjust if your model exposes SR differently
        sil = np.zeros(int(sr * pause_ms / 1000), dtype=np.float32)

        wavs = []
        for i, chunk in enumerate(tqdm(chunks, desc="Sentence to wav: ", position=0, leave=True), 1):
            with torch.inference_mode():
                out = model.inference(
                    text=chunk,
                    language=language,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    enable_text_splitting=False,  # we already split
                    # conservative, stable defaults (tweak as desired)
                    temperature=temperature,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=10.0,
                    length_penalty=1.0,
                    do_sample=True,
                    **hf_generate_kwargs,
                )
            wav = out["wav"] if isinstance(out, dict) and "wav" in out else out
            wav = np.asarray(wav, dtype=np.float32)
            # add a tiny pause between chunks to give prosodic breathing room
            if i > 1:
                wavs.append(sil.copy())
            wavs.append(wav)

        # final assembly with crossfade (covers the pause gap endpoints too)
        wav_full = self._crossfade_concat(wavs, sr, fade_ms=20)
        return chunks, wav_full
