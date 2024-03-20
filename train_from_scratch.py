import os
import subprocess
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)

import torch
import json
from subprocess import Popen
import random
import warnings
import shutil
import logging
from pathlib import Path

from infer.modules.vc.modules import VC
from configs.config import Config

logging.getLogger("numba").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)

config = Config()
vc = VC(config)
ngpu = torch.cuda.device_count()
gpu_infos = []
if_gpu_ok = False

gpus = "-".join([i[0] for i in gpu_infos])


def prepare_dataset(
        exp_dir,
        sr,
        spk_mapping,
        config_path=f"v2/40k.json",
    ):
    cur_dir = Path(__file__).parent
    logger.info(f"Current dir: {cur_dir}")
    # filelist
    gt_wavs_dir = f"{exp_dir}/0_gt_wavs"
    feature_dir = f"{exp_dir}/3_feature768"
    f0_dir = f"{exp_dir}/2a_f0"
    f0nsf_dir = f"{exp_dir}/2b-f0nsf"
    names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
    )
    opt = []
    for name in names:
        # with f0
        opt.append(
            "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
            % (
                gt_wavs_dir.replace("\\", "\\\\"),
                name,
                feature_dir.replace("\\", "\\\\"),
                name,
                f0_dir.replace("\\", "\\\\"),
                name,
                f0nsf_dir.replace("\\", "\\\\"),
                name,
                spk_mapping.get(name, spk_mapping[f"{name.split('_')[0]}"]),
            )
        )

    feature_dim = 768
    # for _ in range(2):
    #     opt.append(
    #         f"{cur_dir}/logs/mute/0_gt_wavs/mute{sr}.wav|{cur_dir}/logs/mute/3_feature{feature_dim}/mute.npy|{cur_dir}/logs/mute/2a_f0/mute.wav.npy|{cur_dir}/logs/mute/2b-f0nsf/mute.wav.npy|0"
    #     )
    random.shuffle(opt)
    Path(f"{exp_dir}/filelist.txt").write_text("\n".join(opt))

    logger.debug("Write filelist done")
    config_save_path = Path(exp_dir) / "config.json"
    if not config_save_path.exists():
        config_save_path.write_text(
            json.dumps(
                config.json_config[config_path],
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
        )


def click_train(
        exp_dir,
        save_dir,
        sr,
        save_every_epoch,
        total_epoch,
        batch_size,
        lr,
        lr_decay,
        if_save_latest,
        pretrained_G,
        pretrained_D,
        gpus,
):
    cur_dir = Path(__file__).parent
    logger.info(f"Current dir: {cur_dir}")

    logger.info("Use gpus: %s", str(gpus))

    cmd = (
        f"PYTHONPATH={cur_dir.parent.absolute()} python {cur_dir}/infer/modules/train/train.py -e {exp_dir} -sr {sr} -f0 1 "
        f"-bs {batch_size} -g {gpus} -te {total_epoch} -se {save_every_epoch} "
        f"--save_dir {save_dir} --lr {lr} --lr_decay {lr_decay} "
        f"{f'-pg {pretrained_G}' if pretrained_G != '' else ''} "
        f"{f'-pd {pretrained_D}' if pretrained_D != '' else ''} "
        f"-l {1 if if_save_latest else 0} -c 0 -sw 0 -v v2"
    )

    logger.info(cmd)
    p = Popen(cmd, shell=True, cwd=cur_dir)
    p.wait()


def full_train(trainset_dir, exp_dir, total_epoch=20, batch_size=8, lr=1.8e-4, lr_decay=0.99):
    # trainset_dir is a folder containing wav files
    per = 3.0 if config.is_half else 3.7
    cur_dir = Path(__file__).parent
    # exp_dir = str(cur_dir / f"logs/{exp_name}")
    save_dir = exp_dir / "weights"
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(exp_dir)
    logger.info("Start training")
    subprocess.run(
        f"PYTHONPATH={cur_dir.parent.absolute()} python {cur_dir}/infer/modules/train/preprocess.py --inp_root {trainset_dir} --sr 48000 --n_p 8 --exp_dir {exp_dir} --per {per}",
        shell=True,
    )

    logger.info("Preprocess done")
    logger.info("Extracting f0 and feature")
    if torch.cuda.is_available():
        subprocess.run(
            f"PYTHONPATH={cur_dir.parent.absolute()} python {cur_dir}/infer/modules/train/extract/extract_f0_rmvpe.py --exp_dir {exp_dir}",
            shell=True,
            check=True,
        )
    else:
        subprocess.run(
            f"PYTHONPATH={cur_dir.parent.absolute()} python {cur_dir}/infer/modules/train/extract/extract_f0_print.py {exp_dir} 8 rmvpe",
            shell=True,
            check=True,
        )

    subprocess.run(
        f"PYTHONPATH={cur_dir.parent.absolute()} python {cur_dir}/infer/modules/train/extract_feature_print.py --exp_dir {exp_dir} --version v2",
        shell=True,
        check=True,
    )

    logger.info("Extracting done")
    click_train(
        exp_dir,
        save_dir,
        "48k",
        True,
        0,
        10,
        total_epoch,
        batch_size,
        lr,
        lr_decay,
        True,
        "models/pretrained_v2/f0G48k.pth",
        "models/pretrained_v2/f0D48k.pth",
        "0",
        True,
        False,
        "v2",
    )


if __name__ == "__main__":
    # full_train("/Users/george/ai-covers-backend/Retrieval-based-Voice-Conversion-gui/data/radiohead_vocals",
    #            "test", 20, 8)
    spk_mapping = json.loads(Path("/home/george/ai-covers-backend/src/rvc/logs/pretrain_1228/speaker_mapping.json").read_text())
    prepare_dataset(exp_dir="logs/pretrain_1228", sr="40k", spk_mapping=spk_mapping, config_path="v2/40k.json")
