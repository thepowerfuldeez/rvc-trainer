import os
import traceback
import argparse
from pathlib import Path
import fairseq
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

# Set environment variables
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=Path, required=True)
    parser.add_argument("--version", type=str, default="v2")
    parser.add_argument("--i_part", type=int, default=0)
    parser.add_argument("--n_part", type=int, default=1)
    return parser.parse_args()

def setup_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    return device

def readwave(wav_path, normalize=False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats

def main():
    args = parse_args()
    device = setup_device()
    exp_dir = args.exp_dir
    version = args.version
    i_part = args.i_part
    n_part = args.n_part

    log_file_path = exp_dir / "extract_f0_feature.log"
    with open(log_file_path, "a+") as f:

        def printt(message):
            print(message)
            f.write(f"{message}\n")
            f.flush()

        model_path = Path(__file__).parents[3] / "models/hubert_base.pt"

        printt(f"Experiment directory: {exp_dir}")
        wavPath = exp_dir / "1_16k_wavs"
        outPath = exp_dir / "3_feature256" if version == "v1" else exp_dir / "3_feature768"
        os.makedirs(outPath, exist_ok=True)

        printt(f"load model(s) from {model_path}")
        if not model_path.exists():
            printt(f"Error: Extracting is shut down because {model_path} does not exist, you may download it from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main")
            exit(0)
        models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [str(model_path)],
            suffix="",
        )
        model = models[0]
        model = model.to(device)
        printt(f"move model to {device}")
        if device not in ["mps", "cpu"]:
            model = model.half()
        model.eval()

        todo = sorted(list(wavPath.iterdir()))[i_part::n_part]
        n = max(1, len(todo) // 10)  # Print at most ten times
        if len(todo) == 0:
            printt("no-feature-todo")
        else:
            printt(f"all-feature-{len(todo)}")
            for idx, file in enumerate(todo):
                try:
                    if file.suffix == ".wav":
                        wav_path = wavPath / file.name
                        out_path = outPath / file.name.replace("wav", "npy")

                        if out_path.exists():
                            continue

                        feats = readwave(wav_path, normalize=saved_cfg.task.normalize)
                        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                        inputs = {
                            "source": feats.half().to(device) if device not in ["mps", "cpu"] else feats.to(device),
                            "padding_mask": padding_mask.to(device),
                            "output_layer": 9 if version == "v1" else 12,
                        }
                        with torch.no_grad():
                            logits = model.extract_features(**inputs)
                            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]

                        feats = feats.squeeze(0).float().cpu().numpy()
                        if np.isnan(feats).sum() == 0:
                            np.save(out_path, feats, allow_pickle=False)
                        else:
                            printt(f"{file.name}-contains nan")
                        if idx % n == 0:
                            printt(f"now-{len(todo)},all-{idx},{file.name},{feats.shape}")
                except:
                    printt(traceback.format_exc())
            printt("all-feature-done")

if __name__ == "__main__":
    main()
