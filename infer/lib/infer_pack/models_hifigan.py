import math
from typing import Optional

import torch
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d, functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

from infer.lib.infer_pack.commons import init_weights
from infer.lib.infer_pack.modules import ResBlock1, ResBlock2, LRELU_SLOPE


# Class representing the generator module in the NSF (Neural Source-Filter) model.
# nsf-hifigan
class GeneratorNSF(torch.nn.Module):
    def __init__(
            self,
            initial_channel,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels,
            sr,
            is_half=False,
    ):
        super(GeneratorNSF, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.f0_upsamp = torch.nn.Upsample(scale_factor=math.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sr, harmonic_num=0, is_half=is_half
        )
        self.noise_convs = nn.ModuleList()
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock = ResBlock1 if resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1:])
                self.noise_convs.append(
                    Conv1d(
                        1,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                    zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        self.upp = math.prod(upsample_rates)

        self.lrelu_slope = LRELU_SLOPE

    def forward(self, x, f0, g: Optional[torch.Tensor] = None):
        har_source, noi_source, uv = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        # torch.jit.script() does not support direct indexing of torch modules
        # That's why I wrote this
        for i, (ups, noise_convs) in enumerate(zip(self.ups, self.noise_convs)):
            if i < self.num_upsamples:
                x = F.leaky_relu(x, self.lrelu_slope)
                x = ups(x)
                x_source = noise_convs(har_source)
                x = x + x_source
                xs: Optional[torch.Tensor] = None
                l = [i * self.num_kernels + j for j in range(self.num_kernels)]
                for j, resblock in enumerate(self.resblocks):
                    if j in l:
                        if xs is None:
                            xs = resblock(x)
                        else:
                            xs += resblock(x)
                # This assertion cannot be ignored! \
                # If ignored, it will cause torch.jit.script() compilation errors
                assert isinstance(xs, torch.Tensor)
                x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()

    def __prepare_scriptable__(self):
        for l in self.ups:
            for hook in l._forward_pre_hooks.values():
                # The hook we want to remove is an instance of WeightNorm class, so
                # normally we would do `if isinstance(...)` but this class is not accessible
                # because of shadowing, so we check the module name directly.
                # https://github.com/pytorch/pytorch/blob/be0ca00c5ce260eb5bcec3237357f7a30cc08983/torch/nn/utils/__init__.py#L3
                if (
                        hook.__module__ == "torch.nn.utils.weight_norm"
                        and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(l)
        for l in self.resblocks:
            for hook in self.resblocks._forward_pre_hooks.values():
                if (
                        hook.__module__ == "torch.nn.utils.weight_norm"
                        and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(l)
        return self


# Class representing the source module for harmonic and noise source filtering.
class SourceModuleHnNSF(torch.nn.Module):
    """SourceModule for NSF-HIFIGAN Vocoder
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(
            self,
            sampling_rate,
            harmonic_num=0,
            sine_amp=0.1,
            add_noise_std=0.003,
            voiced_threshod=0,
            is_half=True,
    ):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.is_half = is_half
        # to produce sine waveforms
        self.l_sin_gen = SineGen(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod
        )

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()
        # self.ddtype:int = -1

    def forward(self, x: torch.Tensor, upp: int = 1):
        sine_wavs, uv, _ = self.l_sin_gen(x, upp)
        sine_wavs = sine_wavs.to(dtype=self.l_linear.weight.dtype)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge, None, None  # noise, uv


# Class for generating sine waves, used in voice synthesis.
class SineGen(torch.nn.Module):
    """Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(torch.pi) or cos(0)
    """

    def __init__(
            self,
            samp_rate,
            harmonic_num=0,
            sine_amp=0.1,
            noise_std=0.003,
            voiced_threshold=0,
            flag_for_pulse=False,
    ):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        if uv.device.type == "privateuseone":  # for DirectML
            uv = uv.float()
        return uv

    def forward(self, f0: torch.Tensor, upp: int):
        """sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        with torch.no_grad():
            f0 = f0[:, None].transpose(1, 2)
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            # fundamental component
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in range(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (
                        idx + 2
                )  # idx + 2: the (idx+1)-th overtone, (idx+2)-th harmonic
            rad_values = (f0_buf / self.sampling_rate) % 1  ###%1意味着n_har的乘积无法后处理优化
            rand_ini = torch.rand(
                f0_buf.shape[0], f0_buf.shape[2], device=f0_buf.device
            )
            rand_ini[:, 0] = 0
            rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
            tmp_over_one = torch.cumsum(rad_values, 1)  # % 1  #####%1意味着后面的cumsum无法再优化
            tmp_over_one *= upp
            tmp_over_one = F.interpolate(
                tmp_over_one.transpose(2, 1),
                scale_factor=float(upp),
                mode="linear",
                align_corners=True,
            ).transpose(2, 1)
            rad_values = F.interpolate(
                rad_values.transpose(2, 1), scale_factor=float(upp), mode="nearest"
            ).transpose(
                2, 1
            )  #######
            tmp_over_one %= 1
            tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
            cumsum_shift = torch.zeros_like(rad_values)
            cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
            sine_waves = torch.sin(
                torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * torch.pi
            )
            sine_waves = sine_waves * self.sine_amp
            uv = self._f02uv(f0)
            uv = F.interpolate(
                uv.transpose(2, 1), scale_factor=float(upp), mode="nearest"
            ).transpose(2, 1)
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)
            sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise
