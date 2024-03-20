import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import argparse
import torch
import traceback
from tqdm import tqdm
from pathlib import Path

from src.whisper.model import Whisper, ModelDimensions
from src.whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--version", type=str, default="v2")
    parser.add_argument("--i_part", type=int, default=0)
    parser.add_argument("--n_part", type=int, default=1)
    return parser.parse_args()


def load_model(path) -> Whisper:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location="cpu")
    dims = ModelDimensions(**checkpoint["dims"])
    print(dims)
    model = Whisper(dims)
    del model.decoder
    cut = len(model.encoder.blocks) // 4
    cut = -1 * cut
    del model.encoder.blocks[cut:]
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    model.half()
    model.to(device)
    return model


def pred_ppg(whisper: Whisper, wav_path, ppg_path):
    audio = load_audio(wav_path)
    audio_length = audio.shape[0]
    ppg_length = audio_length // 320
    audio = pad_or_trim(audio)
    mel = log_mel_spectrogram(audio).half().to(whisper.device)
    with torch.no_grad():
        ppg = whisper.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
        ppg = ppg[:ppg_length, ]  # [length, dim=1280]
        # print(ppg.shape)
        np.save(ppg_path, ppg, allow_pickle=False)


if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    args = parse_args()
    exp_dir = args.exp_dir
    version = args.version
    i_part = args.i_part
    n_part = args.n_part

    input_wavs_path = "%s/1_16k_wavs" % exp_dir
    out_path = "%s/4_ppg1280" % exp_dir
    os.makedirs(out_path, exist_ok=True)
    files = sorted(list(Path(input_wavs_path).glob("**/*.wav")))[i_part::n_part]

    model_path = str(Path(__file__).parent.parent.parent.parent / "models/large-v3.pt")
    whisper = load_model(model_path)

    for wav_path in tqdm(files):
        try:
            save_path = out_path / wav_path.relative_to(input_wavs_path).with_suffix(".npy")
            if os.path.exists(save_path):
                continue
            pred_ppg(whisper, wav_path, save_path)
        except:
            print(traceback.format_exc())
