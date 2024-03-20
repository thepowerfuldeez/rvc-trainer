import os
import sys
import logging
import datetime
import random
import time
from collections import OrderedDict

import torch
import wandb
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

import src.lib.commons as commons
from src.lib.models import RVCModel
from src.lib.discriminator import MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator
from src.utils import get_hparams, get_logger, load_checkpoint, latest_checkpoint_path, \
    save_checkpoint, plot_spectrogram_to_numpy, savee
from src.data_preparation import (
    DistributedBucketSampler,
    TextAudioCollate,
    TextAudioLoader,
)
from src.training_utils import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from src.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

global_step = 0


class EpochRecorder:
    def __init__(self):
        self.last_time = time.time()

    def record(self):
        now_time = time.time()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{current_time}] | ({elapsed_time_str})"


def prepare_data_loaders(hps, n_gpus, rank):
    train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size * n_gpus,
        [100, 200, 300, 400, 500, 600, 700, 800, 900],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    collate_fn = TextAudioCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=16,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )
    return train_loader


def initialize_models_and_optimizers(hps):
    net_g = RVCModel(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
        sr=hps.sample_rate,
    )
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    return net_g, net_d, optim_g, optim_d


def load_model_checkpoint(hps, net_g, net_d, optim_g, optim_d, rank, logger):
    try:
        _, _, _, epoch_str = load_checkpoint(
            latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d, load_opt=hps.enable_opt_load
        )
        if rank == 0:
            logger.info(f"loaded D")
        _, _, _, epoch_str = load_checkpoint(
            latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g, load_opt=hps.enable_opt_load
        )
        global_step = (epoch_str - 1) * len(train_loader)
    except:  # If cannot load, load pretrain
        epoch_str = 1
        global_step = 0
        if hps.pretrainG != "":
            if rank == 0:
                logger.info(f"loaded pretrained {hps.pretrainG}")
            state_dict_g = torch.load(hps.pretrainG, map_location="cpu")["model"]
            if hasattr(net_g, "module"):
                logger.info(
                    net_g.module.load_state_dict(state_dict_g)
                )
            else:
                if hps.enable_opt_load == 0:
                    excluded_keys = {"emb_g.weight"}
                    new_sd = OrderedDict()
                    for k, v in state_dict_g.items():
                        if k not in excluded_keys:
                            new_sd[k] = v
                    state_dict_g = new_sd

                logger.info(
                    net_g.load_state_dict(state_dict_g, strict=False)
                )
        if hps.pretrainD != "":
            if rank == 0:
                logger.info(f"loaded pretrained {hps.pretrainD}")
            state_dict_d = torch.load(hps.pretrainD, map_location="cpu")["model"]
            if hasattr(net_d, "module"):
                logger.info(
                    net_d.module.load_state_dict(state_dict_d)
                )
            else:
                logger.info(
                    net_d.load_state_dict(state_dict_d)
                )
    return epoch_str, global_step


def setup_training(hps, net_g, net_d, optim_g, optim_d, epoch_str):
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    return scheduler_g, scheduler_d


def log_metrics(scalar_dict, image_dict, global_step):
    wandb.log(scalar_dict, step=global_step)
    image_dict_wandb = {k: wandb.Image(v) for k, v in image_dict.items()}
    wandb.log(image_dict_wandb, step=global_step)


def run_training_epoch(epoch, hps, nets, optims, schedulers, train_loader, logger, accelerator):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    global global_step

    net_g.train()
    net_d.train()

    epoch_recorder = EpochRecorder()
    for batch_idx, data in enumerate(train_loader):
        (
            phone,
            phone_lengths,
            pitch,
            pitchf,
            spec,
            spec_lengths,
            wave,
            wave_lengths,
            sid,
            ppg
        ) = data

        model_output = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid,
                             ppg=ppg, enable_perturbation=hps.enable_perturbation == 1)
        y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = model_output

        mel = spec_to_mel_torch(
            spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
        )
        y_mel = commons.slice_segments(
            mel, ids_slice, hps.train.segment_size // hps.data.hop_length
        )

        y_hat_mel = mel_spectrogram_torch(
            y_hat.float().squeeze(1),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
        )
        wave = commons.slice_segments(
            wave, ids_slice * hps.data.hop_length, hps.train.segment_size
        )

        y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
            y_d_hat_r, y_d_hat_g
        )

        optim_d.zero_grad()
        accelerator.backward(loss_disc)
        scheduler_d.step()
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), 1000.0)

        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

        optim_g.zero_grad()
        accelerator.backward(loss_gen_all)
        scheduler_g.step()
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), 1000.0)

        if batch_idx % hps.train.log_interval == 0:
            lr = optim_g.param_groups[0]["lr"]
            logger.info(
                f"Train Epoch: {epoch} [{100.0 * batch_idx / len(train_loader):.0f}%]"
            )
            logger.info([global_step, lr])
            scalar_dict = {
                "loss/total/g": loss_gen_all,
                "loss/total/d": loss_disc,
                "learning_rate": lr,
                "step": global_step,
                "grad_norm_d": grad_norm_d,
                "grad_norm_g": grad_norm_g,
                "loss/g/fm": loss_fm,
                "loss/g/mel": loss_mel,
                "loss/g/kl": loss_kl,
                "loss/g/gen": loss_gen,
            }
            image_dict = {
                "slice/mel_org": plot_spectrogram_to_numpy(
                    y_mel[0].data.cpu().numpy()
                ),
                "slice/mel_gen": plot_spectrogram_to_numpy(
                    y_hat_mel[0].data.cpu().numpy()
                ),
                "all/mel": plot_spectrogram_to_numpy(
                    mel[0].data.cpu().numpy()
                ),
            }

            log_metrics(scalar_dict, image_dict, global_step)

        global_step += 1

    if epoch % hps.save_every_epoch == 0:
        save_checkpoint(
            net_g,
            optim_g,
            hps.train.learning_rate,
            epoch,
            os.path.join(hps.model_dir, f"G_{global_step}.pth"),
        )
        save_checkpoint(
            net_d,
            optim_d,
            hps.train.learning_rate,
            epoch,
            os.path.join(hps.model_dir, f"D_{global_step}.pth"),
        )
        if hps.save_every_weights == "1":
            if hasattr(net_g, "module"):
                ckpt = net_g.module.state_dict()
            else:
                ckpt = net_g.state_dict()
            savee(
                ckpt,
                hps.sample_rate,
                hps.name + "_e%s_s%s" % (epoch, global_step),
                epoch,
                hps.version,
                hps,
            )
            logger.info(f"saving ckpt {hps.name}_e{epoch}:{global_step}")

    logger.info(f"====> Epoch: {epoch} {epoch_recorder.record()}")
    if epoch >= hps.total_epoch or global_step >= hps.train.total_steps:
        logger.info("Training is done. The program is closed.")

        if hasattr(net_g, "module"):
            ckpt = net_g.module.state_dict()
        else:
            ckpt = net_g.state_dict()
        savee(
            ckpt, hps.sample_rate, hps.name, epoch, hps.version, hps
        )
        logger.info(f"saving final ckpt:{hps.name}")
        sleep(1)
        os._exit(2333333)


def run(
        rank,
        n_gpus,
        hps,
        wandb_project_name="coveroke"
):
    global global_step
    accelerator = Accelerator()
    if rank == 0:
        logger = get_logger(hps.model_dir)
        logger.info(hps)
        for _ in range(3):
            try:
                wandb.init(project=wandb_project_name, name=hps.name, config=hps)
                break
            except:
                time.sleep(1)
                pass

    torch.manual_seed(hps.train.seed)

    train_loader = prepare_data_loaders(hps, n_gpus, rank)
    net_g, net_d, optim_g, optim_d = initialize_models_and_optimizers(hps)
    epoch_str, global_step = load_model_checkpoint(hps, net_g, net_d, optim_g, optim_d, rank, logger)
    scheduler_g, scheduler_d = setup_training(hps, net_g, net_d, optim_g, optim_d, epoch_str)

    train_loader, net_g, net_d, optim_g, optim_d, scheduler_g, scheduler_d = accelerator.prepare(
        train_loader, net_g, net_d, optim_g, optim_d, scheduler_g, scheduler_d
    )

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if global_step > hps.train.total_steps:
            break
        run_training_epoch(
            epoch,
            hps,
            [net_g, net_d],
            [optim_g, optim_d],
            [scheduler_g, scheduler_d],
            train_loader,
            logger if rank == 0 else None,
            accelerator,
        )
        scheduler_g.step()
        scheduler_d.step()


if __name__ == "__main__":
    hps = get_hparams()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(random.randint(20000, 55555))

    wandb_project_name = "rvc_train"
    run(rank=0, n_gpus=1, hps=hps, wandb_project_name=wandb_project_name)
