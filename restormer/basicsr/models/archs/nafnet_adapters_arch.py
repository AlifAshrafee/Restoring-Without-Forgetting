import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class Adapter(nn.Module):
    def __init__(self, d_model, bottleneck=64, dropout=0.0, 
                 adapter_scalar="1.0", adapter_layernorm_option="in"):
        super().__init__()
        self.d_model = d_model
        self.bottleneck = bottleneck
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option in ["in", "out"]:  # use LayerNorm on channel dimension
            self.adapter_layer_norm_before = nn.LayerNorm(d_model)

        if adapter_scalar == "learnable_scalar":  # scaling factor
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        # linear layers: [B,C,H,W] → [B,HW,C] → adapter → [B,C,H,W]
        self.down_proj = nn.Linear(d_model, bottleneck)
        self.up_proj = nn.Linear(bottleneck, d_model)

        self.non_linear_func = nn.ReLU()
        self.dropout = dropout

        self._init_lora()

    def _init_lora(self):
        """LoRA-style weights initialization: zero up-projection for identity initialization."""
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual

        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, C)  # [BHW, C]

        if self.adapter_layernorm_option == "in":
            x_flat = self.adapter_layer_norm_before(x_flat)

        down = self.down_proj(x_flat)
        down = self.non_linear_func(down)
        down = F.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        if self.adapter_layernorm_option == "out":
            up = self.adapter_layer_norm_before(up)

        up = up.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        up = up * self.scale

        if add_residual:
            return up + residual
        return up


class AdapterConfig:
    def __init__(
        self,
        bottleneck=64,  # adapter bottleneck dimension
        dropout=0.1,
        adapter_scalar="1.0",  # or 'learnable_scalar'
        adapter_layernorm_option="in",  # 'in', 'out', or None
        adapter_option="parallel",  # 'parallel' or 'sequential'
        adapter_momentum=0.0,  # momentum for weight averaging
    ):
        self.bottleneck = bottleneck
        self.dropout = dropout
        self.adapter_scalar = adapter_scalar
        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_option = adapter_option
        self.adapter_momentum = adapter_momentum


class LayerNormFunction(torch.autograd.Function):
    """Custom LayerNorm operating on channel dimension for [B, C, H, W] tensors."""
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    """Element-wise gating: splits channels in half, multiplies the two halves."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand

        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0, bias=True)

        # Simplified Channel Attention: global average pool → linear → sigmoid
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp, adapter=None, adapter_option="parallel"):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta  # post-attention residual

        if adapter is not None:
            adapt_x = adapter(y, add_residual=False)
        else:
            adapt_x = None

        residual = y
        ffn_out = self.conv5(self.sg(self.conv4(self.norm2(y))))
        ffn_out = self.dropout2(ffn_out)
        ffn_out = ffn_out * self.gamma

        if adapt_x is not None:
            if adapter_option == "sequential":
                out = residual + ffn_out
                out = adapter(out)  # adapter wraps the combined output
            elif adapter_option == "parallel":
                out = residual + ffn_out + adapt_x
            else:
                raise ValueError(f"Unknown adapter_option: {adapter_option}")
        else:
            out = residual + ffn_out

        return out


class NAFNetAdapters(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1,
                 enc_blk_nums=[], dec_blk_nums=[], adapter_config=None):
        super().__init__()

        if adapter_config is None:
            self.adapter_config = AdapterConfig()
        elif isinstance(adapter_config, dict):
            self.adapter_config = AdapterConfig(**adapter_config)
        else:
            self.adapter_config = adapter_config

        self.width = width
        self.middle_blk_num = middle_blk_num
        self.enc_blk_nums = list(enc_blk_nums)
        self.dec_blk_nums = list(dec_blk_nums)

        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.ModuleList([NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2

        self.middle_blks = nn.ModuleList([NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2))
            )
            chan //= 2
            self.decoders.append(nn.ModuleList([NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

        self.total_blocks = self._count_blocks()
        self.block_dims = self._get_block_dims()

        self.cur_adapter = nn.ModuleList()    # current domain adapter set
        self.adapter_list = nn.ModuleList()   # list of all previously committed domain adapter sets

        self.init_adapters()  # initialize adapter set for first domain

    def _count_blocks(self):
        return sum(self.enc_blk_nums) + self.middle_blk_num + sum(self.dec_blk_nums)

    def _get_block_dims(self):
        dims = []
        chan = self.width

        # encoder: blocks operate at chan, then downsample to 2*chan
        for num in self.enc_blk_nums:
            dims.extend([chan] * num)
            chan *= 2

        # middle: blocks at the deepest channel width
        dims.extend([chan] * self.middle_blk_num)

        # decoder: upsample first (chan → chan//2), then blocks at chan//2
        for num in self.dec_blk_nums:
            chan //= 2
            dims.extend([chan] * num)

        return dims

    def _make_adapter_set(self):
        config = self.adapter_config
        adapter_set = nn.ModuleList()

        for dim in self.block_dims:
            adapter_set.append(
                Adapter(
                    d_model=dim,
                    bottleneck=config.bottleneck,
                    dropout=config.dropout,
                    adapter_scalar=config.adapter_scalar,
                    adapter_layernorm_option=config.adapter_layernorm_option,
                )
            )

        try:  # matching device of existing model parameters
            device = next(self.parameters()).device
            adapter_set = adapter_set.to(device)
        except StopIteration:
            pass  # during __init__, no parameters yet

        return adapter_set

    def init_adapters(self):
        self.cur_adapter = self._make_adapter_set()
        self.cur_adapter.requires_grad_(True)

    def commit_and_reinit(self):
        """
        Moves cur_adapter into adapter_list (frozen storage)
        and re-initializes a fresh LoRA-zero cur_adapter for the next domain.
        """
        self.adapter_list.append(copy.deepcopy(self.cur_adapter))
        self.init_adapters()  # reinitializes cur_adapter with LoRA zero-init

    def prepare_adapter_list_for_loading(self, state_dict):
        adapter_list_indices = set()
        for key in state_dict.keys():
            if key.startswith("adapter_list."):
                idx = int(key.split(".")[1])
                adapter_list_indices.add(idx)

        num_committed = len(adapter_list_indices)
        if num_committed > 0:
            print(f"Checkpoint contains {num_committed} committed adapter(s), pre-allocating adapter_list")
            for _ in range(num_committed):
                self.adapter_list.append(self._make_adapter_set())

    def freeze_backbone(self):
        for name, param in self.named_parameters():
            if "cur_adapter" in name:
                param.requires_grad = True  # only current domain adapter trains
            else:
                param.requires_grad = False  # backbone + all adapter_list entries frozen

        print(f"Backbone frozen.")
        print(f"Trainable adapter parameters in current adapter set:")
        print(f"{sum(p.numel() for p in self.cur_adapter.parameters()):,}")

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

    def forward(self, inp, adapter_id=-1):
        """
        adapter_id=-1: backbone only, no adapter
        adapter_id=len(adapter_list): cur_adapter (current training domain)
        adapter_id=k < len(adapter_list): frozen committed adapter for domain k
        """
        adapter_option = self.adapter_config.adapter_option

        if adapter_id == -1:  # no adapters, pure backbone
            adapters = None
        elif adapter_id == len(self.adapter_list):  # N: current adapter
            adapters = self.cur_adapter
        elif 0 <= adapter_id < len(self.adapter_list):  # 0~N-1: previous adapters
            adapters = self.adapter_list[adapter_id]
        else:
            raise ValueError(f"Invalid adapter_id={adapter_id}."
                             f"adapter_list has {len(self.adapter_list)} committed adapter(s), "
                             f"cur_adapter is at index {len(self.adapter_list)}.")

        adapter_idx = 0

        def get_adapter():
            nonlocal adapter_idx
            if adapters is None:
                return None
            a = adapters[adapter_idx]
            adapter_idx += 1
            return a

        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            for blk in encoder:
                x = blk(x, get_adapter(), adapter_option)
            encs.append(x)
            x = down(x)

        for blk in self.middle_blks:
            x = blk(x, get_adapter(), adapter_option)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            for blk in decoder:
                x = blk(x, get_adapter(), adapter_option)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]
