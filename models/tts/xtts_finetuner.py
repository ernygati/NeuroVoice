import os
import torch
from pathlib import Path
from huggingface_hub import snapshot_download
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from alltalk_tts.trainer_alltalk.trainer import Trainer, TrainerArgs

OUTPUT_SAMPLE_RATE = 24000
class XTTSFinetuner:
    def __init__(self, save_checkpoint_path = "./models/tts/finetuned/xtts"):
        self.output_path = Path(save_checkpoint_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def load_base_model(self, hug_repo_name = "coqui/XTTS-v2", local_folder = "./models/tts/base"):
        save_path = os.path.join(local_folder, hug_repo_name)
        snapshot_download(
            repo_id=hug_repo_name,
            local_dir=save_path,
            local_dir_use_symlinks=False,  # Avoid symlinks (better for direct file access)
            revision="main"  # Branch/tag name (default: "main")
        )
        return save_path
        
    
    def finetune_model(self, device,base_model_path, train_metadata, eval_metadata, 
                       language="ru", num_epochs=10, batch_size=4, grad_accum=1, 
                       learning_rate=5e-6, checkpoint_pth_path=None,output_dir=None):
        """
        Finetune XTTS model on prepared dataset.
        
        Args:
            base_model_path: Path to pretrained XTTS model directory
            train_metadata: Path to training metadata CSV
            eval_metadata: Path to evaluation metadata CSV
            language: Language code
            num_epochs: Number of training epochs
            batch_size: Batch size
            grad_accum: Gradient accumulation steps
            learning_rate: Initial learning rate
            output_dir: Directory to save finetuned model
            
        Returns:
            Path to finetuned model checkpoint
        """
        # Setup paths
        base_model_path = Path(base_model_path)
        output_dir = Path(output_dir) if output_dir else self.output_path / "finetuned_model"
        output_dir.mkdir(exist_ok=True)
        
        # Load dataset
        config_dataset = BaseDatasetConfig(
            formatter="coqui",
            dataset_name="ft_dataset",
            path=train_metadata.parent,
            meta_file_train=train_metadata.name,
            meta_file_val=eval_metadata.name,
            language=language,
        )
        
        # Model configuration
        if os.path.exists(str(checkpoint_pth_path)):
            print(f"Starting from existing checkpoint {str(checkpoint_pth_path)} ...")
            checkpoint_pth_path = str(checkpoint_pth_path)
        else:
            print(f"Starting from base checpoint {str(base_model_path / 'model.pth')} ...")
            checkpoint_pth_path = str(base_model_path / "model.pth")

        model_args = GPTArgs(
            max_conditioning_length=132300,
            min_conditioning_length=66150,
            max_wav_length=int(11 * 22050),  # ~11 seconds
            max_text_length=200,
            mel_norm_file=str(base_model_path / "mel_stats.pth"),
            dvae_checkpoint=str(base_model_path / "dvae.pth"),
            xtts_checkpoint=checkpoint_pth_path,
            tokenizer_file=str(base_model_path / "vocab.json"),
            gpt_num_audio_tokens=1026,
            gpt_start_audio_token=1024,
            gpt_stop_audio_token=1025,
            gpt_use_masking_gt_prompt_approach=True,
            gpt_use_perceiver_resampler=True,
        )
        
        audio_config = XttsAudioConfig(
            sample_rate=22050,
            dvae_sample_rate=22050,
            output_sample_rate=OUTPUT_SAMPLE_RATE
        )
        
        # Training configuration
        config = GPTTrainerConfig(
            epochs=num_epochs,
            output_path=output_dir,
            model_args=model_args,
            run_name="xtts finetune",
            project_name="XTTS_trainer",
            run_description="GPT XTTS training",
            dashboard_logger="tensorboard",
            audio=audio_config,
            batch_size=batch_size,
            batch_group_size=48,
            eval_batch_size=batch_size,
            num_loader_workers=2,
            eval_split_max_size=256,
            print_step=50,
            plot_step=100,
            log_model_step=100,
            save_step=1000,
            save_n_checkpoints=1,
            save_checkpoints=True,
            print_eval=False,
            # Optimizer values like tortoise, pytorch implementation with
            # modifications to not apply WD to non-weight parameters.
            optimizer="AdamW",
            optimizer_wd_only_on_weights=True,
            optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
            lr=learning_rate,  # learning rate
            lr_scheduler="CosineAnnealingWarmRestarts",
            # it was adjusted accordly for the new step scheme
            lr_scheduler_params={'T_0': num_epochs//4, 'eta_min': 1e-6},
            test_sentences=[]
        )
        
        # Initialize and train model
        model = GPTTrainer.init_from_config(config)
        train_samples, eval_samples = load_tts_samples([config_dataset], eval_split=True)
        
        trainer = Trainer(
            TrainerArgs(gpu=device.index),
            config,
            output_path=output_dir,
            model=model,
            train_samples=train_samples,
            eval_samples=eval_samples,
            gpu=device.index
        )
        
        trainer.fit()
        
        # Return path to best model checkpoint
        return output_dir
    
    def load_model(self, device,config_path, checkpoint_path, vocab_path,speaker_xtts_path):
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
        
        if torch.cuda.is_available():
            model.to(device)
            
        return model
    
    def generate_audio(self, model, text, reference_audio, language="ru"):
        """Generate audio from text using the finetuned model."""
        # Get conditioning latents from reference audio
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[reference_audio] if isinstance(reference_audio, str) else reference_audio,
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

# Example usage:
if __name__ == "__main__":
    finetuner = XTTSFinetuner()