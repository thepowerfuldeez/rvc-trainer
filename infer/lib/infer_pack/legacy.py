import math
from typing import Optional

import torch
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d, functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

from rvc.infer.lib.infer_pack import attentions, commons, modules
from rvc.infer.lib.infer_pack.commons import init_weights
from rvc.infer.lib.infer_pack.models import sr2sr, PosteriorEncoder, logger, \
    TextEncoder768
from infer.lib.infer_pack.models_hifigan import GeneratorNSF
from rvc.infer.lib.infer_pack.modules import ResidualCouplingBlock


# Probably not used
class TextEncoder256(nn.Module):
    def __init__(
            self,
            out_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            f0=True,
    ):
        super(TextEncoder256, self).__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.emb_phone = nn.Linear(256, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        if f0 == True:
            self.emb_pitch = nn.Embedding(256, hidden_channels)  # pitch 256
        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
            self, phone: torch.Tensor, pitch: Optional[torch.Tensor], lengths: torch.Tensor
    ):
        if pitch is None:
            x = self.emb_phone(phone)
        else:
            x = self.emb_phone(phone) + self.emb_pitch(pitch)
        x = x * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = self.lrelu(x)
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask


class SynthesizerTrnMs256NSFsid(nn.Module):
    def __init__(
            self,
            spec_channels,
            segment_size,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            spk_embed_dim,
            gin_channels,
            sr,
            **kwargs
    ):
        super(SynthesizerTrnMs256NSFsid, self).__init__()
        if isinstance(sr, str):
            sr = sr2sr[sr]
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        # self.hop_length = hop_length#
        self.spk_embed_dim = spk_embed_dim
        self.enc_p = TextEncoder256(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
        )
        self.dec = GeneratorNSF(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            sr=sr,
            is_half=kwargs["is_half"],
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels
        )
        self.emb_g = nn.Embedding(self.spk_embed_dim, gin_channels)
        logger.debug(
            "gin_channels: "
            + str(gin_channels)
            + ", self.spk_embed_dim: "
            + str(self.spk_embed_dim)
        )

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        if hasattr(self, "enc_q"):
            self.enc_q.remove_weight_norm()

    def __prepare_scriptable__(self):
        for hook in self.dec._forward_pre_hooks.values():
            # The hook we want to remove is an instance of WeightNorm class, so
            # normally we would do `if isinstance(...)` but this class is not accessible
            # because of shadowing, so we check the module name directly.
            # https://github.com/pytorch/pytorch/blob/be0ca00c5ce260eb5bcec3237357f7a30cc08983/torch/nn/utils/__init__.py#L3
            if (
                    hook.__module__ == "torch.nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
            ):
                torch.nn.utils.remove_weight_norm(self.dec)
        for hook in self.flow._forward_pre_hooks.values():
            if (
                    hook.__module__ == "torch.nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
            ):
                torch.nn.utils.remove_weight_norm(self.flow)
        if hasattr(self, "enc_q"):
            for hook in self.enc_q._forward_pre_hooks.values():
                if (
                        hook.__module__ == "torch.nn.utils.weight_norm"
                        and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(self.enc_q)
        return self

    @torch.jit.ignore
    def forward(
            self,
            phone: torch.Tensor,
            phone_lengths: torch.Tensor,
            pitch: torch.Tensor,
            pitchf: torch.Tensor,
            y: torch.Tensor,
            y_lengths: torch.Tensor,
            ds: Optional[torch.Tensor] = None,
    ):  # 这里ds是id，[bs,1]
        # print(1,pitch.shape)#[bs,t]
        g = self.emb_g(ds).unsqueeze(-1)  # [b, 256, 1]##1是t，广播的
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        # print(-1,pitchf.shape,ids_slice,self.segment_size,self.hop_length,self.segment_size//self.hop_length)
        pitchf = commons.slice_segments2(pitchf, ids_slice, self.segment_size)
        # print(-2,pitchf.shape,z_slice.shape)
        o = self.dec(z_slice, pitchf, g=g)
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    @torch.jit.export
    def infer(
            self,
            phone: torch.Tensor,
            phone_lengths: torch.Tensor,
            pitch: torch.Tensor,
            nsff0: torch.Tensor,
            sid: torch.Tensor,
            skip_head: Optional[torch.Tensor] = None,
            return_length: Optional[torch.Tensor] = None,
    ):
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
        if skip_head is not None and return_length is not None:
            assert isinstance(skip_head, torch.Tensor)
            assert isinstance(return_length, torch.Tensor)
            head = int(skip_head.item())
            length = int(return_length.item())
            z_p = z_p[:, :, head: head + length]
            x_mask = x_mask[:, :, head: head + length]
            nsff0 = nsff0[:, head: head + length]
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec(z * x_mask, nsff0, g=g)
        return o, x_mask, (z, z_p, m_p, logs_p)


class SynthesizerTrnMs256NSFsid_nono(nn.Module):
    def __init__(
            self,
            spec_channels,
            segment_size,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            spk_embed_dim,
            gin_channels,
            sr=None,
            **kwargs
    ):
        super(SynthesizerTrnMs256NSFsid_nono, self).__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        # self.hop_length = hop_length#
        self.spk_embed_dim = spk_embed_dim
        self.enc_p = TextEncoder256(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
            f0=False,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels
        )
        self.emb_g = nn.Embedding(self.spk_embed_dim, gin_channels)
        logger.debug(
            "gin_channels: "
            + str(gin_channels)
            + ", self.spk_embed_dim: "
            + str(self.spk_embed_dim)
        )

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.enc_q.remove_weight_norm()

    def __prepare_scriptable__(self):
        for hook in self.dec._forward_pre_hooks.values():
            # The hook we want to remove is an instance of WeightNorm class, so
            # normally we would do `if isinstance(...)` but this class is not accessible
            # because of shadowing, so we check the module name directly.
            # https://github.com/pytorch/pytorch/blob/be0ca00c5ce260eb5bcec3237357f7a30cc08983/torch/nn/utils/__init__.py#L3
            if (
                    hook.__module__ == "torch.nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
            ):
                torch.nn.utils.remove_weight_norm(self.dec)
        for hook in self.flow._forward_pre_hooks.values():
            if (
                    hook.__module__ == "torch.nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
            ):
                torch.nn.utils.remove_weight_norm(self.flow)
        if hasattr(self, "enc_q"):
            for hook in self.enc_q._forward_pre_hooks.values():
                if (
                        hook.__module__ == "torch.nn.utils.weight_norm"
                        and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(self.enc_q)
        return self

    @torch.jit.ignore
    def forward(self, phone, phone_lengths, y, y_lengths, ds):  # 这里ds是id，[bs,1]
        g = self.emb_g(ds).unsqueeze(-1)  # [b, 256, 1]##1是t，广播的
        m_p, logs_p, x_mask = self.enc_p(phone, None, phone_lengths)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        o = self.dec(z_slice, g=g)
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    @torch.jit.export
    def infer(
            self,
            phone: torch.Tensor,
            phone_lengths: torch.Tensor,
            sid: torch.Tensor,
            rate: Optional[torch.Tensor] = None,
    ):
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, None, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
        if rate is not None:
            head = int(z_p.shape[2] * (1.0 - rate.item()))
            z_p = z_p[:, :, head:]
            x_mask = x_mask[:, :, head:]
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec(z * x_mask, g=g)
        return o, x_mask, (z, z_p, m_p, logs_p)


class SynthesizerTrnMs768NSFsid_nono(nn.Module):
    def __init__(
            self,
            spec_channels,
            segment_size,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            spk_embed_dim,
            gin_channels,
            sr=None,
            **kwargs
    ):
        super(SynthesizerTrnMs768NSFsid_nono, self).__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        # self.hop_length = hop_length#
        self.spk_embed_dim = spk_embed_dim
        self.enc_p = TextEncoder768(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
            f0=False,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels
        )
        self.emb_g = nn.Embedding(self.spk_embed_dim, gin_channels)
        logger.debug(
            "gin_channels: "
            + str(gin_channels)
            + ", self.spk_embed_dim: "
            + str(self.spk_embed_dim)
        )

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.enc_q.remove_weight_norm()

    def __prepare_scriptable__(self):
        for hook in self.dec._forward_pre_hooks.values():
            # The hook we want to remove is an instance of WeightNorm class, so
            # normally we would do `if isinstance(...)` but this class is not accessible
            # because of shadowing, so we check the module name directly.
            # https://github.com/pytorch/pytorch/blob/be0ca00c5ce260eb5bcec3237357f7a30cc08983/torch/nn/utils/__init__.py#L3
            if (
                    hook.__module__ == "torch.nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
            ):
                torch.nn.utils.remove_weight_norm(self.dec)
        for hook in self.flow._forward_pre_hooks.values():
            if (
                    hook.__module__ == "torch.nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
            ):
                torch.nn.utils.remove_weight_norm(self.flow)
        if hasattr(self, "enc_q"):
            for hook in self.enc_q._forward_pre_hooks.values():
                if (
                        hook.__module__ == "torch.nn.utils.weight_norm"
                        and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(self.enc_q)
        return self

    @torch.jit.ignore
    def forward(self, phone, phone_lengths, y, y_lengths, ds):  # 这里ds是id，[bs,1]
        g = self.emb_g(ds).unsqueeze(-1)  # [b, 256, 1]##1是t，广播的
        m_p, logs_p, x_mask = self.enc_p(phone, None, phone_lengths)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        o = self.dec(z_slice, g=g)
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    @torch.jit.export
    def infer(
            self,
            phone: torch.Tensor,
            phone_lengths: torch.Tensor,
            sid: torch.Tensor,
            rate: Optional[torch.Tensor] = None,
    ):
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, None, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
        if rate is not None:
            head = int(z_p.shape[2] * (1.0 - rate.item()))
            z_p = z_p[:, :, head:]
            x_mask = x_mask[:, :, head:]
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec(z * x_mask, g=g)
        return o, x_mask, (z, z_p, m_p, logs_p)


class Generator(torch.nn.Module):
    def __init__(
            self,
            initial_channel,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=0,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
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

    def forward(self, x: torch.Tensor, g: Optional[torch.Tensor] = None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

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
            for hook in l._forward_pre_hooks.values():
                if (
                        hook.__module__ == "torch.nn.utils.weight_norm"
                        and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(l)
        return self

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
