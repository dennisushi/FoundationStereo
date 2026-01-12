import logging
import math
import os
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["EdgeNeXt", "build_edgenext_small"]

_LOGGER = logging.getLogger(__name__)
_DEFAULT_WEIGHTS = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "weights", "edgenext_small_timm.bin"
)


def trunc_normal_tf_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)
    return tensor


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


def calculate_drop_path_rates(drop_path_rate: float, depths: Sequence[int]) -> List[List[float]]:
    if drop_path_rate == 0.0:
        return [[0.0] * d for d in depths]
    total = sum(depths)
    rates = torch.linspace(0, drop_path_rate, total, dtype=torch.float32).tolist()
    out: List[List[float]] = []
    idx = 0
    for depth in depths:
        out.append(rates[idx : idx + depth])
        idx += depth
    return out


class LayerNorm2d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(dim=1, keepdim=True)
        s = (x - u).pow(2).mean(dim=1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def create_conv2d(
    in_chs: int,
    out_chs: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: Optional[int] = None,
    depthwise: bool = False,
    bias: bool = True,
):
    if padding is None:
        padding = kernel_size // 2
    groups = in_chs if depthwise else 1
    return nn.Conv2d(in_chs, out_chs, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)


def _pool2d(x: torch.Tensor, pool_type: str) -> torch.Tensor:
    if not pool_type:
        return x
    if pool_type == "avg":
        return x.mean(dim=(-2, -1), keepdim=True)
    if pool_type == "max":
        return x.amax(dim=(-2, -1), keepdim=True)
    if pool_type == "avgmax":
        return 0.5 * (x.mean(dim=(-2, -1), keepdim=True) + x.amax(dim=(-2, -1), keepdim=True))
    raise ValueError(f"Unsupported pool_type={pool_type}")


class ClassifierHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, pool_type: str = "avg", drop_rate: float = 0.0):
        super().__init__()
        self.pool_type = pool_type
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(in_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = _pool2d(x, self.pool_type)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        if pre_logits:
            return x
        return self.fc(x)

    def reset(self, num_classes: int, pool_type: Optional[str] = None):
        if pool_type is not None:
            self.pool_type = pool_type
        in_features = self.fc.in_features if isinstance(self.fc, nn.Linear) else num_classes
        self.fc = nn.Linear(in_features, num_classes) if num_classes > 0 else nn.Identity()


class NormMlpClassifierHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        pool_type: str = "avg",
        drop_rate: float = 0.0,
        norm_layer: Callable[[int], nn.Module] = LayerNorm2d,
        hidden_size: Optional[int] = None,
        act_layer: Callable[[], nn.Module] = nn.Tanh,
    ):
        super().__init__()
        self.pool_type = pool_type
        self.norm = norm_layer(in_features)
        self.pre_logits: nn.Module
        if hidden_size:
            self.pre_logits = nn.Sequential(
                nn.Linear(in_features, hidden_size),
                act_layer(),
            )
            self.num_features = hidden_size
        else:
            self.pre_logits = nn.Identity()
            self.num_features = in_features
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = _pool2d(x, self.pool_type)
        x = self.norm(x)
        x = torch.flatten(x, 1)
        x = self.pre_logits(x)
        x = self.drop(x)
        if pre_logits:
            return x
        return self.fc(x)

    def reset(self, num_classes: int, pool_type: Optional[str] = None):
        if pool_type is not None:
            self.pool_type = pool_type
        self.fc = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()


def named_apply(fn: Callable[[nn.Module, str], None], module: nn.Module, name: str = ""):
    fn(module, name)
    for child_name, child in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name
        named_apply(fn, child, full_name)


class PositionalEncodingFourier(nn.Module):
    def __init__(self, hidden_dim: int = 32, dim: int = 768, temperature: float = 10000.0):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, shape: Tuple[int, int, int]) -> torch.Tensor:
        b, h, w = shape
        weight = self.token_projection.weight
        device = weight.device
        dtype = weight.dtype
        mask = torch.zeros((b, h, w), device=device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos.to(dtype))
        return pos


class ConvBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        kernel_size: int = 7,
        stride: int = 1,
        conv_bias: bool = True,
        expand_ratio: float = 4.0,
        ls_init_value: float = 1e-6,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        act_layer: Callable[[], nn.Module] = nn.GELU,
        drop_path: float = 0.0,
    ):
        super().__init__()
        dim_out = dim_out or dim
        self.shortcut_after_dw = stride > 1 or dim != dim_out

        self.conv_dw = create_conv2d(dim, dim_out, kernel_size=kernel_size, stride=stride, depthwise=True, bias=conv_bias)
        self.norm = norm_layer(dim_out)
        self.mlp = Mlp(dim_out, int(expand_ratio * dim_out), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim_out)) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv_dw(x)
        if self.shortcut_after_dw:
            shortcut = x

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        return shortcut + self.drop_path(x)


class CrossCovarianceAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, -1).permute(2, 0, 3, 4, 1)
        q, k, v = qkv.unbind(0)
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)) * self.temperature
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = attn @ v
        x = x.permute(0, 3, 1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def no_weight_decay(self):
        return {"temperature"}


class SplitTransposeBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_scales: int = 1,
        num_heads: int = 8,
        expand_ratio: float = 4.0,
        use_pos_emb: bool = True,
        conv_bias: bool = True,
        qkv_bias: bool = True,
        ls_init_value: float = 1e-6,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        act_layer: Callable[[], nn.Module] = nn.GELU,
        drop_path: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        width = max(int(math.ceil(dim / num_scales)), int(math.floor(dim // num_scales)))
        self.num_scales = max(1, num_scales - 1)

        convs = []
        for _ in range(self.num_scales):
            convs.append(create_conv2d(width, width, kernel_size=3, depthwise=True, bias=conv_bias))
        self.convs = nn.ModuleList(convs)

        self.pos_embd = PositionalEncodingFourier(dim=dim) if use_pos_emb else None
        self.norm_xca = norm_layer(dim)
        self.gamma_xca = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value > 0 else None
        self.xca = CrossCovarianceAttn(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.norm = norm_layer(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(expand_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        spx = x.chunk(len(self.convs) + 1, dim=1)
        spo = []
        sp = spx[0]
        for i, conv in enumerate(self.convs):
            if i > 0:
                sp = sp + spx[i]
            sp = conv(sp)
            spo.append(sp)
        spo.append(spx[-1])
        x = torch.cat(spo, 1)

        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).permute(0, 2, 1)
        if self.pos_embd is not None:
            pos_encoding = self.pos_embd((b, h, w)).reshape(b, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding.to(x.dtype)
        if self.gamma_xca is not None:
            x = x + self.drop_path(self.gamma_xca * self.xca(self.norm_xca(x)))
        else:
            x = x + self.drop_path(self.xca(self.norm_xca(x)))
        x = x.reshape(b, h, w, c)

        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        return shortcut + self.drop_path(x)


class EdgeNeXtStage(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        stride: int = 2,
        depth: int = 2,
        num_global_blocks: int = 1,
        num_heads: int = 4,
        scales: int = 2,
        kernel_size: int = 7,
        expand_ratio: float = 4.0,
        use_pos_emb: bool = False,
        downsample_block: bool = False,
        conv_bias: bool = True,
        ls_init_value: float = 1.0,
        drop_path_rates: Optional[List[float]] = None,
        norm_layer: Callable[..., nn.Module] = LayerNorm2d,
        norm_layer_cl: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        act_layer: Callable[[], nn.Module] = nn.GELU,
    ):
        super().__init__()
        drop_path_rates = drop_path_rates or [0.0] * depth
        if downsample_block or stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                nn.Conv2d(in_chs, out_chs, kernel_size=2, stride=2, bias=conv_bias),
            )
            in_chs = out_chs

        stage_blocks = []
        for i in range(depth):
            if i < depth - num_global_blocks:
                stage_blocks.append(
                    ConvBlock(
                        dim=in_chs,
                        dim_out=out_chs,
                        stride=stride if downsample_block and i == 0 else 1,
                        conv_bias=conv_bias,
                        kernel_size=kernel_size,
                        expand_ratio=expand_ratio,
                        ls_init_value=ls_init_value,
                        drop_path=drop_path_rates[i],
                        norm_layer=norm_layer_cl,
                        act_layer=act_layer,
                    )
                )
            else:
                stage_blocks.append(
                    SplitTransposeBlock(
                        dim=in_chs,
                        num_scales=scales,
                        num_heads=num_heads,
                        expand_ratio=expand_ratio,
                        use_pos_emb=use_pos_emb,
                        conv_bias=conv_bias,
                        ls_init_value=ls_init_value,
                        drop_path=drop_path_rates[i],
                        norm_layer=norm_layer_cl,
                        act_layer=act_layer,
                    )
                )
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class EdgeNeXt(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: str = "avg",
        dims: Tuple[int, ...] = (24, 48, 88, 168),
        depths: Tuple[int, ...] = (3, 3, 9, 3),
        global_block_counts: Tuple[int, ...] = (0, 1, 1, 1),
        kernel_sizes: Tuple[int, ...] = (3, 5, 7, 9),
        heads: Tuple[int, ...] = (8, 8, 8, 8),
        d2_scales: Tuple[int, ...] = (2, 2, 3, 4),
        use_pos_emb: Tuple[bool, ...] = (False, True, False, False),
        ls_init_value: float = 1e-6,
        head_init_scale: float = 1.0,
        expand_ratio: float = 4.0,
        downsample_block: bool = False,
        conv_bias: bool = True,
        stem_type: str = "patch",
        head_norm_first: bool = False,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        drop_path_rate: float = 0.0,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.drop_rate = drop_rate
        norm_layer = partial(LayerNorm2d, eps=1e-6)
        norm_layer_cl = partial(nn.LayerNorm, eps=1e-6)

        if stem_type == "patch":
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4, bias=conv_bias),
                norm_layer(dims[0]),
            )
        elif stem_type == "overlap":
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=9, stride=4, padding=9 // 2, bias=conv_bias),
                norm_layer(dims[0]),
            )
        else:
            raise ValueError(f"Unsupported stem_type={stem_type}")

        dp_rates = calculate_drop_path_rates(drop_path_rate, depths)
        curr_stride = 4
        stages = []
        in_chs = dims[0]
        for i in range(4):
            stride = 2 if curr_stride == 2 or i > 0 else 1
            curr_stride *= stride
            stages.append(
                EdgeNeXtStage(
                    in_chs=in_chs,
                    out_chs=dims[i],
                    stride=stride,
                    depth=depths[i],
                    num_global_blocks=global_block_counts[i],
                    num_heads=heads[i],
                    drop_path_rates=dp_rates[i],
                    scales=d2_scales[i],
                    expand_ratio=expand_ratio,
                    kernel_size=kernel_sizes[i],
                    use_pos_emb=use_pos_emb[i],
                    ls_init_value=ls_init_value,
                    downsample_block=downsample_block,
                    conv_bias=conv_bias,
                    norm_layer=norm_layer,
                    norm_layer_cl=norm_layer_cl,
                    act_layer=act_layer,
                )
            )
            in_chs = dims[i]
        self.stages = nn.Sequential(*stages)

        self.num_features = dims[-1]
        if head_norm_first:
            self.norm_pre = norm_layer(self.num_features)
            self.head = ClassifierHead(
                self.num_features,
                num_classes,
                pool_type=global_pool,
                drop_rate=self.drop_rate,
            )
        else:
            self.norm_pre = nn.Identity()
            self.head = NormMlpClassifierHead(
                self.num_features,
                num_classes,
                pool_type=global_pool,
                drop_rate=self.drop_rate,
                norm_layer=norm_layer,
            )

        named_apply(lambda module, name: _init_weights(module, name, head_init_scale), self)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm_pre(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        return self.head(x, pre_logits=pre_logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _init_weights(module: nn.Module, name: Optional[str] = None, head_init_scale: float = 1.0):
    if isinstance(module, nn.Conv2d):
        trunc_normal_tf_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        trunc_normal_tf_(module.weight, std=0.02)
        nn.init.zeros_(module.bias)
        if name and name.endswith(".head.fc"):
            module.weight.data.mul_(head_init_scale)
            module.bias.data.mul_(head_init_scale)


def _load_state_dict(model: nn.Module, checkpoint: dict):
    state_dict = checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        _LOGGER.warning("Missing keys when loading EdgeNeXt weights: %s", missing)
    if unexpected:
        _LOGGER.warning("Unexpected keys when loading EdgeNeXt weights: %s", unexpected)


def build_edgenext_small(
    pretrained: bool = True,
    weights_path: Optional[str] = None,
    map_location: str = "cpu",
    **kwargs,
) -> EdgeNeXt:
    model = EdgeNeXt(
        depths=(3, 3, 9, 3),
        dims=(48, 96, 160, 304),
        global_block_counts=(0, 1, 1, 1),
        kernel_sizes=(3, 5, 7, 9),
        d2_scales=(2, 2, 3, 4),
        use_pos_emb=(False, True, False, False),
        drop_path_rate=0.1,
        **kwargs,
    )
    if pretrained:
        ckpt_path = weights_path or _DEFAULT_WEIGHTS
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"EdgeNeXt weights not found at {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=map_location)
        _load_state_dict(model, checkpoint)
    return model
