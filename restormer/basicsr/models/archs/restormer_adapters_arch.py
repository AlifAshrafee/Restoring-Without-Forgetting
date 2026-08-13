import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers


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


class OverlapPatchEmbed(nn.Module):
    """
    Overlapped image patch embedding using a 3x3 convolution with stride 1 and padding 1.
    This allows for overlapping patches, which can help capture local context better than non-overlapping.
    """
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    """
    Gated-Dconv Feed-Forward Network (GDFN) with depth-wise convolution and gating mechanism.
    """
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)  # [B, C, H, W] → [B, hidden_features*2, H, W]
        x1, x2 = self.dwconv(x).chunk(2, dim=1)  # split into two, each: [B, hidden_features, H, W]
        x = F.gelu(x1) * x2  # gating: GELU(x1) ⊙ x2
        x = self.project_out(x)  # [B, hidden_features, H, W] → [B, C, H, W]
        return x


class Attention(nn.Module):
    """
    Multi-DConv Head Transposed Self-Attention (MDTA) with transposed attention mechanism.
    """
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkv projection: single conv, then split into q, k, v
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # depth-wise convolution on q, k, v to introduce local context
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        _, _, h, w = x.shape

        # generate Q, K, V with local context using depth-wise convolution
        qkv = self.qkv_dwconv(self.qkv(x))  # [B, 3C, H, W]
        q, k, v = qkv.chunk(3, dim=1)  # each: [B, C, H, W]

        # reshape for multi-head attention: [B, C, H, W] → [B, num_heads, C//num_heads, H*W]
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        # normalize Q and K for stable attention scores
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # transposed attention: C×C instead of N×N
        # this allows attention to be computed across channels for each spatial location, 
        # rather than across spatial locations for each channel.
        # q: [B, num_heads, C//num_heads, H*W], k: [B, num_heads, C//num_heads, H*W]
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # [B, num_heads, C//num_heads, C//num_heads]
        attn = attn.softmax(dim=-1)

        # apply attention to values
        out = attn @ v  # [B, num_heads, C//num_heads, H*W]

        # reshape back to [B, C, H, W]
        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    """
    Restormer Transformer Block with adapter integration.
    """

    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, adapter=None, adapter_option="parallel"):
        x = x + self.attn(self.norm1(x))

        if adapter is not None:
            adapt_x = adapter(x, add_residual=False)
        else:
            adapt_x = None

        residual = x
        ffn_out = self.ffn(self.norm2(x))

        if adapt_x is not None:
            if adapter_option == "sequential":  # adapter applied after FFN output: FFN → Adapter
                x = residual + ffn_out
                x = adapter(x)
            elif adapter_option == "parallel":  # adapter output added in parallel with FFN: FFN + Adapter
                x = residual + ffn_out + adapt_x
            else:
                raise ValueError(f"Unknown adapter_option: {adapter_option}")
        else:
            x = residual + ffn_out

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            # spatial downsampling by rearranging pixels into channels: [B, C, H, W] → [B, C*4, H//2, W//2]
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            # channel upsampling by rearranging channels into pixels: [B, C, H, W] → [B, C//4, H*2, W*2]
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class RestormerAdapters(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias", # other option 'BiasFree'
        adapter_config=None,
    ):
        super().__init__()

        if adapter_config is None:
            self.adapter_config = AdapterConfig()
        elif isinstance(adapter_config, dict):
            self.adapter_config = AdapterConfig(**adapter_config)
        else:
            self.adapter_config = adapter_config

        self.level_dims = [
            dim,  # level 1
            int(dim * 2**1),  # level 2 (96)
            int(dim * 2**2),  # level 3 (192)
            int(dim * 2**3),  # level 4 / latent (384)
        ]
        self.num_blocks = num_blocks
        self.num_refinement_blocks = num_refinement_blocks

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.down1_2 = Downsample(dim)  # from level 1 to level 2 (48 → 96)

        self.encoder_level2 = nn.ModuleList(
            [
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.down2_3 = Downsample(int(dim * 2**1))  # from level 2 to level 3 (96 → 192)

        self.encoder_level3 = nn.ModuleList(
            [
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[2])
            ]
        )

        self.down3_4 = Downsample(int(dim * 2**2))  # from level 3 to level 4 (192 → 384)

        self.latent = nn.ModuleList(
            [
                TransformerBlock(
                    dim=int(dim * 2**3),
                    num_heads=heads[3],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[3])
            ]
        )

        self.up4_3 = Upsample(int(dim * 2**3))  # from level 4 to level 3 (384 → 192)
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias
        )

        self.decoder_level3 = nn.ModuleList(
            [
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(int(dim * 2**2))  # from level 3 to level 2 (192 → 96)
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias
        )

        self.decoder_level2 = nn.ModuleList(
            [
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(int(dim * 2**1))  # from level 2 to level 1 (no 1x1 conv to reduce channels)

        self.decoder_level1 = nn.ModuleList(
            [
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.refinement = nn.ModuleList(
            [
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_refinement_blocks)
            ]
        )

        self.output = nn.Conv2d(
            int(dim * 2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

        self.total_blocks = self._count_blocks()  # total number of transformer blocks
        self.block_dims = self._get_block_dims()

        self.cur_adapter = nn.ModuleList()  # current domain adapter set
        self.adapter_list = nn.ModuleList()  # list of all previously committed domain adapter sets

        self.init_adapters()  # initialize adapter set for first domain

    def _count_blocks(self):
        n = self.num_blocks
        # encoder (4 levels) + decoder (3 levels, level1 uses dim*2) + refinement
        return (n[0] + n[1] + n[2] + n[3]  # encoder
                + n[2] + n[1] + n[0]  # decoder
                + self.num_refinement_blocks)  # refinement

    def _get_block_dims(self):
        dims = []
        n = self.num_blocks

        # encoder levels
        dims.extend([self.level_dims[0]] * n[0])  # level 1
        dims.extend([self.level_dims[1]] * n[1])  # level 2
        dims.extend([self.level_dims[2]] * n[2])  # level 3
        dims.extend([self.level_dims[3]] * n[3])  # latent

        # decoder levels
        dims.extend([self.level_dims[2]] * n[2])  # level 3
        dims.extend([self.level_dims[1]] * n[1])  # level 2
        dims.extend([self.level_dims[1]] * n[0])  # level 1 (dim*2 after concat)

        # refinement
        dims.extend([self.level_dims[1]] * self.num_refinement_blocks)

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

    def forward(self, inp_img, adapter_id=-1):
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

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = inp_enc_level1
        for blk in self.encoder_level1:
            out_enc_level1 = blk(out_enc_level1, get_adapter(), adapter_option)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = inp_enc_level2
        for blk in self.encoder_level2:
            out_enc_level2 = blk(out_enc_level2, get_adapter(), adapter_option)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = inp_enc_level3
        for blk in self.encoder_level3:
            out_enc_level3 = blk(out_enc_level3, get_adapter(), adapter_option)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = inp_enc_level4
        for blk in self.latent:
            latent = blk(latent, get_adapter(), adapter_option)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = inp_dec_level3
        for blk in self.decoder_level3:
            out_dec_level3 = blk(out_dec_level3, get_adapter(), adapter_option)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = inp_dec_level2
        for blk in self.decoder_level2:
            out_dec_level2 = blk(out_dec_level2, get_adapter(), adapter_option)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = inp_dec_level1
        for blk in self.decoder_level1:
            out_dec_level1 = blk(out_dec_level1, get_adapter(), adapter_option)

        for blk in self.refinement:
            out_dec_level1 = blk(out_dec_level1, get_adapter(), adapter_option)

        out = self.output(out_dec_level1) + inp_img

        return out
