import os
import torch
import torchaudio
from tqdm import tqdm

class VADProcessor:
    def __init__(self,device, min_duration, max_duration, vad_folder="./models/VAD",max_gap=0.5):
        self._min_duration=min_duration
        self._max_duration =max_duration
        self._vad_folder=vad_folder
        self._device = device
        self._max_gap = max_gap

    def load_vad_model(self):
            """Load Silero VAD model with optimized settings"""
            os.makedirs(self._vad_folder, exist_ok=True)
            torch.hub.set_dir(self._vad_folder)
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                verbose=False
            )
            vad_model = model.to(self._device)
            get_speech_timestamps, collect_chunks = utils[0], utils[4]
            return vad_model, get_speech_timestamps, collect_chunks

    def _resample_wav_to_16KHz(self,wav, sr):
        sr_16k = 16000
        wav_16k = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=sr_16k).to(wav.device)
        return wav_16k, sr_16k
        

    def _process_audio_with_vad(self,wav_16k, sr, vad_model, get_speech_timestamps, min_duration=2.0):
        """
        Enhanced VAD processing with dynamic parameters
        Args:
            wav: Audio tensor (1D) in 16kHz
            sr: Original sample rate
            vad_model: Loaded VAD model
            get_speech_timestamps: VAD utility function
            min_duration: Minimum segment duration in seconds
        """
        # Dynamic VAD parameters based on minimum duration
        min_speech_ms = max(200, int(min_duration * 500))  # At least 200ms but proportional to min_duration
        min_silence_ms = 300 if min_duration < 3 else 500  # Longer silence for longer segments
        
        vad_segments = get_speech_timestamps(
            wav_16k,
            vad_model,
            sampling_rate=16000,
            threshold=0.2,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
            window_size_samples=1024,
            speech_pad_ms=300
        )
        
        # return merge_vad_segments(vad_segments, min_duration=min_duration)
        return vad_segments

    def _segments_backscale_and_padding(self,vad_segments, sr):
        # Scale timestamps back to original sample rate
        scale_factor = sr / 16000
        for segment in vad_segments:
            segment["start"] = int(segment["start"] * scale_factor)
            segment["end"] = int(segment["end"] * scale_factor) + int(0.2 * sr)  # Add 200ms padding
        return vad_segments


    def _merge_short_segments(self, segments,sr, min_duration,max_duration, max_gap=0.5):
        """
        More aggressive merge strategy for short segments
        - Increases max_gap to 0.5s (from 0.3s)
        - Looks ahead multiple segments for potential merges
        - Considers surrounding context
        """
        if not segments:
            return segments

        merged = []
        current_group = []
        target_duration = ((min_duration + 10.0) / 2)*sr  # Target middle of range
        max_gap = max_gap*sr
        for i, segment in enumerate(segments):
            current_duration = sum(s["end"] - s["start"] 
                                for s in current_group) if current_group else 0

            # If this is a continuation of current group
            # print(f"i:{i}, (segment['start'] - current_group[-1]['end']) {(segment['start'] - current_group[-1]['end']) if current_group else None} ,  max_gap:{max_gap}")
            if current_group and (segment["start"] - current_group[-1]["end"]) <= max_gap:
                # Check if adding this segment gets us closer to target duration
                new_duration = current_duration + (segment["end"] - segment["start"])
                # print(f"i:{i}, new_duration:{new_duration}, abs(new_duration - target_duration) {abs(new_duration - target_duration)/sr} < abs(current_duration - target_duration) {abs(current_duration - target_duration)/sr}?")
                if abs(new_duration - target_duration) < abs(current_duration - target_duration):
                    current_group.append(segment)
                else:
                    # Save current group and start new one
                    merged_segment = {
                        "start": current_group[0]["start"],
                        "end": current_group[-1]["end"]
                    }
                    merged.append(merged_segment)
                    current_group = [segment]
            else:
                # Save previous group if it exists
                if current_group:
                    merged_segment = {
                        "start": current_group[0]["start"],
                        "end": current_group[-1]["end"]
                    }
                    merged.append(merged_segment)
                current_group = [segment]
                if i==2:
                    break

        # Handle last group
        if current_group:
            merged_segment = {
                "start": current_group[0]["start"],
                "end": current_group[-1]["end"]
            }
            merged.append(merged_segment)
        print(f"VAD segments num before merging: {len(segments)} and after: {len(merged)}")
    
        return merged

    def get_vad_segments(self, wav, sr) -> list[dict]:
        vad_model, get_speech_timestamps, collect_chunks= self.load_vad_model()
        wav_16k, sr_16k = self._resample_wav_to_16KHz(wav, sr)
        vad_segments = self._process_audio_with_vad(wav_16k, sr, vad_model, get_speech_timestamps)
        vad_segments = self._segments_backscale_and_padding(vad_segments, sr)
        vad_merged_segments = self._merge_short_segments(vad_segments , sr, min_duration=self._min_duration,max_duration=self._max_duration, max_gap=self._max_gap)
        return vad_merged_segments