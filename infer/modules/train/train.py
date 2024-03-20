import os
import sys
import logging
import datetime
from collections import OrderedDict
from pathlib import Path
from random import randint, shuffle
from time import sleep
from time import time as ttime

import torch
import wandb
import torch.profiler
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

logger = logging.getLogger(__name__)

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

from rvc.infer.lib.train.utils import get_hparams, get_logger, load_checkpoint, latest_checkpoint_path, \
    save_checkpoint, plot_spectrogram_to_numpy

hps = get_hparams()
os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")
n_gpus = len(hps.gpus.split("-"))

from rvc.infer.lib.infer_pack import commons
from rvc.infer.lib.train.data_utils import (
    DistributedBucketSampler,
    TextAudioCollate,
    TextAudioCollateMultiNSFsid,
    TextAudioLoader,
    TextAudioLoaderMultiNSFsid,
)
from rvc.infer.lib.infer_pack.discriminator import MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator
from rvc.infer.lib.infer_pack.models import (
    SynthesizerTrnMs768NSFsid as RVC_Model_f0,
)

from rvc.infer.lib.train.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from rvc.infer.lib.train.stft_loss import MultiResolutionSTFTLoss
from rvc.infer.lib.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from rvc.infer.lib.train.process_ckpt import savee

global_step = 0


class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{current_time}] | ({elapsed_time_str})"


def main():
    n_gpus = torch.cuda.device_count()

    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        n_gpus = 1
    if n_gpus < 1:
        # patch to unblock people without gpus. there is probably a better way.
        print("NO GPU DETECTED: falling back to CPU - this may take a while")
        n_gpus = 1
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))

    wandb_project_name = "rvc_train"

    children = []
    for i in range(n_gpus):
        subproc = mp.Process(
            target=run,
            args=(i, n_gpus, hps, wandb_project_name),
        )
        children.append(subproc)
        subproc.start()

    for i in range(n_gpus):
        children[i].join()


def run(
        rank,
        n_gpus,
        hps,
        wandb_project_name="coveroke"
):
    global global_step
    if rank == 0:
        logger = get_logger(hps.model_dir)
        logger.info(hps)
        wandb.init(project=wandb_project_name, name=hps.name, config=hps)

    dist.init_process_group(
        backend="gloo", init_method="env://", world_size=n_gpus, rank=rank
    )
    torch.manual_seed(hps.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    # --------
    # DATA INIT
    # --------
    train_dataset = TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size * n_gpus,
        # [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200,1400],  # 16s
        [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    # It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.
    # num_workers=8 -> num_workers=4
    if hps.if_f0 == 1:
        collate_fn = TextAudioCollateMultiNSFsid()
    else:
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
    # --------
    # DATA INIT
    # --------

    # MODEL INIT
    net_g = RVC_Model_f0(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
        is_half=hps.train.fp16_run,
        sr=hps.sample_rate,
    )
    if torch.cuda.is_available():
        net_g = net_g.cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    if torch.cuda.is_available():
        net_d = net_d.cuda(rank)

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
    
    # resume training
    try: 
        _, _, _, epoch_str = load_checkpoint(
            latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d, load_opt=hps.enable_opt_load
        )
        if rank == 0:
            logger.info("loaded D")
        _, _, _, epoch_str = load_checkpoint(
            latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g, load_opt=hps.enable_opt_load
        )
        global_step = (epoch_str - 1) * len(train_loader)
        # epoch_str = 1
        # global_step = 0
    except:  # 如果首次不能加载，加载pretrain
        # traceback.print_exc()
        epoch_str = 1
        global_step = 0
        if hps.pretrainG != "":
            if rank == 0:
                logger.info("loaded pretrained %s" % (hps.pretrainG))
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
                logger.info("loaded pretrained %s" % (hps.pretrainD))
            state_dict_d = torch.load(hps.pretrainD, map_location="cpu")["model"]
            if hasattr(net_d, "module"):
                logger.info(
                    net_d.module.load_state_dict(state_dict_d)
                )
            else:
                logger.info(
                    net_d.load_state_dict(state_dict_d)
                )
    # MODEL INIT

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    device = "cuda"
    resolutions = [(1024, 120, 600), (2048, 240, 1200), (4096, 480, 2400), (512, 50, 240)]
    # stft_criterion = MultiResolutionSTFTLoss(device, resolutions)

    # with torch.profiler.profile(
    #         activities=[torch.profiler.ProfilerActivity.CPU,
    #                     torch.profiler.ProfilerActivity.CUDA],
    #         schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=2),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler(str(Path(hps.experiment_dir) / 'profiler_logs/')),
    #         record_shapes=True,
    #         profile_memory=True,
    #         with_stack=True) as profiler:
    cache = []
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if global_step > hps.train.total_steps:
            break
        train_and_evaluate(
            rank,
            epoch,
            hps,
            [net_g, net_d],
            [optim_g, optim_d],
            [scheduler_g, scheduler_d],
            scaler,
            [train_loader, None],
            logger if rank == 0 else None,
            None,
            cache,
            # stft_criterion=stft_criterion,
            # profiler=profiler,
        )
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(
        rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, cache, stft_criterion=None,
        profiler=None
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()

    clip_value = 10.0

    # Run steps
    epoch_recorder = EpochRecorder()
    for batch_idx, data in enumerate(train_loader):
        ## Unpack
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

        ## Load on CUDA
        if not hps.if_cache_data_in_gpu and torch.cuda.is_available():
            phone = phone.cuda(rank, non_blocking=True)
            phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
            if hps.if_ppg == 1:
                ppg = ppg.cuda(rank, non_blocking=True)
            if hps.if_f0 == 1:
                pitch = pitch.cuda(rank, non_blocking=True)
                pitchf = pitchf.cuda(rank, non_blocking=True)
            sid = sid.cuda(rank, non_blocking=True)
            spec = spec.cuda(rank, non_blocking=True)
            spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
            wave = wave.cuda(rank, non_blocking=True)
            # wave_lengths = wave_lengths.cuda(rank, non_blocking=True)

        # Calculate
        with autocast(enabled=hps.train.fp16_run):
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

            with autocast(enabled=False):
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
            if hps.train.fp16_run:
                y_hat_mel = y_hat_mel.half()
            wave = commons.slice_segments(
                wave, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )

        optim_d.zero_grad()
        scaler.scale(loss_disc).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), clip_value)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel

                # Multi-Resolution STFT Loss
                # sc_loss, mag_loss = stft_criterion(y_hat.float().squeeze(1), wave.squeeze(1))
                # stft_loss = (sc_loss + mag_loss) * hps.train.c_stft

                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl #+ stft_loss
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), clip_value)
        scaler.step(optim_g)
        scaler.update()
        if profiler is not None:
            profiler.step()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                # Amor For Tensorboard display
                if loss_mel > 75:
                    loss_mel = 75
                if loss_kl > 9:
                    loss_kl = 9

                logger.info([global_step, lr])
                logger.info(
                    f"loss_disc={loss_disc:.3f}, loss_gen={loss_gen:.3f}, loss_fm={loss_fm:.3f},loss_mel={loss_mel:.3f}, loss_kl={loss_kl:.3f}"
                )
                scalar_dict = {
                    "loss/total/g": loss_gen_all,
                    "loss/total/d": loss_disc,
                    "learning_rate": lr,
                    "step": global_step,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        # "loss/g/stft": stft_loss,
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/kl": loss_kl,
                    }
                )

                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                )
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                )
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

                wandb.log(
                    scalar_dict,
                    step=global_step,
                )
                image_dict_wandb = {
                    k: wandb.Image(v)
                    for k, v in image_dict.items()
                }
                wandb.log(
                    image_dict_wandb,
                    step=global_step,
                )

        global_step += 1
    # /Run steps

    if epoch % hps.save_every_epoch == 0 and rank == 0:
        if hps.if_latest == 0:
            save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
            )
            save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
            )
        else:
            save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "G_{}.pth".format(2333333)),
            )
            save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "D_{}.pth".format(2333333)),
            )
        if rank == 0 and hps.save_every_weights == "1":
            if hasattr(net_g, "module"):
                ckpt = net_g.module.state_dict()
            else:
                ckpt = net_g.state_dict()
            logger.info(
                "saving ckpt %s_e%s:%s"
                % (
                    hps.name,
                    epoch,
                    savee(
                        ckpt,
                        hps.sample_rate,
                        hps.if_f0,
                        hps.name + "_e%s_s%s" % (epoch, global_step),
                        epoch,
                        hps.version,
                        hps,
                    ),
                )
            )

    if rank == 0:
        logger.info("====> Epoch: {} {}".format(epoch, epoch_recorder.record()))
    if epoch >= hps.total_epoch and rank == 0 or global_step >= hps.train.total_steps and rank == 0:
        logger.info("Training is done. The program is closed.")

        if hasattr(net_g, "module"):
            ckpt = net_g.module.state_dict()
        else:
            ckpt = net_g.state_dict()
        logger.info(
            "saving final ckpt:%s"
            % (
                savee(
                    ckpt, hps.sample_rate, hps.if_f0, hps.name, epoch, hps.version, hps
                )
            )
        )
        sleep(1)
        os._exit(2333333)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
