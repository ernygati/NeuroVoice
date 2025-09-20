import os, platform, torch
torch.backends.quantized.engine = "qnnpack"
from torch import nn
from torch.ao.quantization import quantize_dynamic
from transformers.pytorch_utils import Conv1D  

# --------------------------
# 0) CPU knobs (set once)
# -------------------------
class Quantizer:
    def __init__(self,model):
        self._model=model 
    def _select_quant_engine(self):
        # Apple Silicon / ARM -> qnnpack; x86 -> fbgemm
        is_arm = platform.machine().lower() in ("arm64", "aarch64")
        return "qnnpack" if is_arm else "fbgemm"
    
    def setup_cpu_runtime(self,num_threads=None):
        torch.backends.quantized.engine = _select_quant_engine()
        if num_threads is None:
            num_threads = max(1, (os.cpu_count() or 4))
        torch.set_num_threads(num_threads)                   # intra-op
        torch.set_num_interop_threads(max(1, num_threads//2))# inter-op
        os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
        print(f"[cpu] engine={torch.backends.quantized.engine}, threads={num_threads}/{max(1, num_threads//2)}")
    
    # ---------------------------------------------
    # 1) Convert HF Conv1D -> nn.Linear in GPT2Model
    # ---------------------------------------------
    def replace_hf_conv1d_with_linear(self,module):
        """
        Recursively replace HuggingFace transformers.pytorch_utils.Conv1D
        blocks with nn.Linear so dynamic quantization can catch them.
        Safe to run multiple times; skips if already Linear.
        """
    
        for name, child in list(module.named_children()):
            # Identify by type OR by best-effort name match if HF isn't importable
            is_hf_conv1d = (Conv1D is not None and isinstance(child, Conv1D)) \
                        or (child.__class__.__name__ == "Conv1D")
    
            if is_hf_conv1d:
                W = child.weight        # shape (in_features, out_features) in HF
                b = child.bias
                out_f = b.numel()
                in_f  = W.numel() // out_f
                linear = nn.Linear(in_f, out_f, bias=True)
                with torch.no_grad():
                    # HF Conv1D stores weight as (in, out); nn.Linear expects (out, in)
                    linear.weight.copy_(W.view(in_f, out_f).t())
                    linear.bias.copy_(b)
                setattr(module, name, linear)
                print(f"[patch] {module.__class__.__name__}.{name}: Conv1D -> nn.Linear ({in_f}->{out_f})")
    
            else:
                self.replace_hf_conv1d_with_linear(child)
    
    # -----------------------------------------------------
    # 2) Quantize GPT (and optionally speaker_encoder.fc)
    # -----------------------------------------------------
    def quantize_xtts_for_cpu(self, quantize_speaker_fc: bool = True):
        self._model.eval()
    
        # a) Patch GPT2Model inside your GPT wrapper
        if hasattr(self._model, "gpt") and hasattr(self._model.gpt, "gpt"):
            gpt2 = self._model.gpt.gpt
            self.replace_hf_conv1d_with_linear(gpt2)
    
            # b) Dynamic-quantize the transformer block (Linear-heavy)
            q_gpt2 = quantize_dynamic(gpt2, {nn.Linear}, dtype=torch.qint8)
            self._model.gpt.gpt = q_gpt2
            print("[quant] gpt.gpt (GPT2Model) -> dynamic int8")
    
        else:
            print("[warn] Could not find self._model.gpt.gpt (GPT2Model). No GPT quantization applied.")
        
        #c) Quntize gpt_infrerence block
        if hasattr(self._model.gpt, "gpt_inference"):
            gpt_inf = self._model.gpt.gpt_inference.transformer
            self._model.gpt.gpt_inference.transformer = quantize_dynamic(
                gpt_inf, {nn.Linear}, dtype=torch.qint8
            )
            print("[quant] gpt.gpt_inference (GPT2InferenceModel) -> dynamic int8")
        
        # Heads (!DOES NOT WORK on MAC)
        # if isinstance(getattr(self._model.gpt, "text_head", None), nn.Linear):
        #     self._model.gpt.text_head = quantize_dynamic(self._model.gpt.text_head, {nn.Linear}, dtype=torch.qint8)
        #     print("[quant] gpt.text_head -> int8")
        # if isinstance(getattr(self._model.gpt, "mel_head", None), nn.Linear):
        #     self._model.gpt.mel_head = quantize_dynamic(self._model.gpt.mel_head, {nn.Linear}, dtype=torch.qint8)
        #     print("[quant] gpt.mel_head -> int8")
        # if hasattr(self._model.gpt, "gpt_inference") and isinstance(getattr(self._model.gpt.gpt_inference, "lm_head", None), nn.Sequential):
        #     if isinstance(self._model.gpt.gpt_inference.lm_head[1], nn.Linear):
        #         self._model.gpt.gpt_inference.lm_head[1] = quantize_dynamic(
        #             self._model.gpt.gpt_inference.lm_head[1], {nn.Linear}, dtype=torch.qint8
        #         )
                # print("[quant] gpt.gpt_inference.lm_head[1] -> int8")
        # d) Optional: quantize speaker encoder final FC (tiny but free)
        if quantize_speaker_fc and hasattr(self._model, "hifigan_decoder"):
            se = getattr(self._model.hifigan_decoder, "speaker_encoder", None)
            if se is not None and hasattr(se, "fc") and isinstance(se.fc, nn.Linear):
                se.fc = quantize_dynamic(se.fc, {nn.Linear}, dtype=torch.qint8)
                print("[quant] speaker_encoder.fc -> dynamic int8")
    
        return self._model
    
    # -----------------------------------------------------
    # 3) (Optional) Try TorchScript to trim Python overhead
    # -----------------------------------------------------
    def maybe_torchscript(self, save_path: str = "xtts_v2_cpu_int8.pt"):
        try:
            with torch.inference_mode():
                scripted = torch.jit.script(self._model)
                torch.jit.optimize_for_inference(scripted)
                torch.jit.save(scripted, save_path)
            print(f"[jit] Saved TorchScript: {save_path}")
            return scripted
        except Exception as e:
            print(f"[jit] Skipped TorchScript: {e}")
            return self._model
    
    # -----------------------------------------------------
    # 4) Glue it together for your instance
    # -----------------------------------------------------
    def accelerate(self, torchscript: bool = False, num_threads: int | None = None):
        # setup_cpu_runtime(num_threads=num_threads)
        self._model = self.quantize_xtts_for_cpu(quantize_speaker_fc=True)
        if torchscript:
            self._model = self.maybe_torchscript()
        return self._model
    
    # --------------- Example usage ----------------
if __name__=="__main__":
    quantizer=Quantizer(model)
    model = quantizer.accelerate(torchscript=False)
    # wav = self._model.inference(text="Привет! Это тест.", language="ru", speaker_wav="path/to/ref.wav")