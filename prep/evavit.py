from typing import Tuple, Callable, Optional, List, Union
import math

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn import LayerNorm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist

from einops import rearrange

from timm.layers import PatchDropout, trunc_normal_, apply_keep_indices_nlc, \
    RotaryEmbeddingCat, use_fused_attn, apply_rot_embed_cat, DropPath, SwiGLU, GluMlp, Mlp
from batchgenerators.utilities.file_and_folder_operations import load_json

from nnssl.training.nnsslTrainer.evaMAE.dynamic_network.vit_embed_decode import PatchEmbed, PatchDecode, LayerNormNd

def resample_abs_pos_embed_3d(
        posemb,
        new_size: List[int],
        old_size: Optional[List[int]] = None,
        num_prefix_tokens: int = 1,
        interpolation: str = 'bicubic',
        antialias: bool = True,
        verbose: bool = False,
):
    raise NotImplementedError('This still needs to be adjusted for 3d')
    # sort out sizes, assume square if old size not provided
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return posemb

    if old_size is None:
        hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
        old_size = hw, hw

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.float()  # interpolate needs float32
    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    posemb = posemb.to(orig_dtype)

    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb], dim=1)

    return posemb


class EvaAttention(nn.Module):
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            num_prefix_tokens: int = 1,
            qkv_bias_separate: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            attn_head_dim: Optional[int] = None,
            norm_layer: Optional[Callable] = None,
    ):
        """

        Args:
            dim:
            num_heads:
            qkv_bias:
            qkv_fused:
            attn_drop:
            proj_drop:
            attn_head_dim:
            norm_layer:
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5
        self.num_prefix_tokens = num_prefix_tokens
        self.fused_attn = use_fused_attn()
        self.qkv_bias_separate = qkv_bias_separate

        if qkv_fused:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
            self.q_proj = self.k_proj = self.v_proj = None
            if qkv_bias:
                self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
                self.register_buffer('k_bias', torch.zeros(all_head_dim), persistent=False)
                self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
            else:
                self.q_bias = self.k_bias = self.v_bias = None
        else:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.qkv = None
            self.q_bias = self.k_bias = self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(all_head_dim) if norm_layer is not None else nn.Identity()
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x,
            rope: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        B, N, C = x.shape

        if self.qkv is not None:
            if self.q_bias is None:
                qkv = self.qkv(x)
            else:
                qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
                if self.qkv_bias_separate:
                    qkv = self.qkv(x)
                    qkv += qkv_bias
                else:
                    qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        else:
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # B, num_heads, N, C
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

        if rope is not None:
            npt = self.num_prefix_tokens
            q = torch.cat([q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope)], dim=2).type_as(v)
            k = torch.cat([k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope)], dim=2).type_as(v)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            if attn_mask is not None:
                attn_mask = attn_mask.to(torch.bool)
                attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
            attn = attn.softmax(dim=-1)

            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EvaBlock(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            num_prefix_tokens: int = 1,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            init_values: Optional[float] = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            attn_head_dim: Optional[int] = None,
            drop_path_scale: bool = True
    ):
        """

        Args:
            dim:
            num_heads:
            qkv_bias:
            qkv_fused:
            mlp_ratio:
            swiglu_mlp:
            scale_mlp:
            scale_attn_inner:
            proj_drop:
            attn_drop:
            drop_path:
            init_values:
            act_layer:
            norm_layer:
            attn_head_dim:
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = EvaAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            num_prefix_tokens=num_prefix_tokens,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer if scale_attn_inner else None,
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path1 = DropPath(drop_path, drop_path_scale) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        hidden_features = int(dim * mlp_ratio)
        if swiglu_mlp:
            if scale_mlp:
                # when norm in SwiGLU used, an impl with separate fc for gate & x is used
                self.mlp = SwiGLU(
                    in_features=dim,
                    hidden_features=hidden_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                )
            else:
                # w/o any extra norm, an impl with packed weights is used, matches existing GluMLP
                self.mlp = GluMlp(
                    in_features=dim,
                    hidden_features=hidden_features * 2,
                    norm_layer=norm_layer if scale_mlp else None,
                    act_layer=nn.SiLU,
                    gate_last=False,
                    drop=proj_drop,
                )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=hidden_features,
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path2 = DropPath(drop_path, drop_path_scale) if drop_path > 0. else nn.Identity()

    def forward(self, x, rope: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None):
        if self.gamma_1 is None:
            x = x + self.drop_path1(self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class Eva(nn.Module):
    """ Eva Vision Transformer w/ Abs & Rotary Pos Embed

    This class implements the EVA and EVA02 models that were based on the BEiT ViT variant
      * EVA - abs pos embed, global avg pool
      * EVA02 - abs + rope pos embed, global avg pool, SwiGLU, scale Norm in MLP (ala normformer)


    """

    def __init__(
            self,
            embed_dim: int = 1024,
            depth: int = 24,
            num_heads: int = 16,
            qkv_bias: bool = True,
            qkv_fused: bool = False,
            mlp_ratio: float = 4 * 2 / 3,
            swiglu_mlp: bool = True,
            scale_mlp: bool = True,
            scale_attn_inner: bool = False,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,  # drops input patches, may be used for MAE style pretraining
            proj_drop_rate: float = 0.,  # drops out things related to the projection. That is in the MLP and at the end of EVA attention
            attn_drop_rate: float = 0.,  # drops attention, meaning connections between patches may bebroken up at random
            drop_path_rate: float = 0.,  # drops computations (multihead attention, mlp), Implementation of scaling might be useless here because this is not batch normed
            norm_layer: Callable = LayerNorm,
            init_values: Optional[float] = None,
            class_token: bool = False,
            use_abs_pos_emb: bool = True,
            use_rot_pos_emb: bool = True,
            dynamic_img_size: bool = False,
            ref_feat_shape: Optional[Tuple[int, ...]] = None,# 224/14
            num_reg_tokens: int = 0,
            drop_path_scale: bool = True,
            rope_impl = RotaryEmbeddingCat,
            rope_kwargs = None
    ):
        """
        Diff to timm implementation

        - removed patch embedding, we expect embeded patches
        - removed classification token, we use features at the end
        - removed head
        - dynamic image size is not supported, but left in for future stuff
        - self.cls_token removed
        - removed postnorm block support
        """
        super().__init__()
        if rope_kwargs is None:
            rope_kwargs = {}

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        self.num_prefix_tokens = (1 if class_token else 0) + num_reg_tokens

        num_patches = np.prod(ref_feat_shape)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, num_reg_tokens, embed_dim)) if num_reg_tokens else None
        self.cls_embed = class_token and self.reg_token is None

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_prefix_tokens, embed_dim)) if use_abs_pos_emb else None
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
                return_indices=True,
            )
        else:
            self.patch_drop = None

        if use_rot_pos_emb:
            # self.rope = VisionRotaryEmbeddingFast_Fabian3D(
            #     embed_dim // num_heads,
            #     ft_seq_len=ref_feat_shape
            # )
            if len(ref_feat_shape) == 3:
                rope_dim = round(embed_dim // num_heads / 1.5)
                assert rope_dim == embed_dim / num_heads / 1.5, 'rope dim must be divsible by (num_heads * 1.5)'
                assert rope_dim % 4 == 0, 'rope dim must be divisible by 4'
            else:
                rope_dim = embed_dim // num_heads
            self.rope = rope_impl(
                rope_dim,
                in_pixels=False,
                feat_shape=ref_feat_shape,
                ref_feat_shape=ref_feat_shape,
                **rope_kwargs
            )
        else:
            self.rope = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        block_fn = EvaBlock
        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                mlp_ratio=mlp_ratio,
                swiglu_mlp=swiglu_mlp,
                scale_mlp=scale_mlp,
                scale_attn_inner=scale_attn_inner,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                num_prefix_tokens=self.num_prefix_tokens,
                drop_path_scale=drop_path_scale
            )
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        self.apply(self._init_weights)
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)
        if self.reg_token is not None:
            trunc_normal_(self.reg_token, std=.02)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {'pos_embed', 'cls_token'}
        return nwd

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))],
        )
        return matcher

    def _pos_embed(self, x) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.dynamic_img_size:
            raise NotImplementedError('dynamic_img_size is not implemented at the moment')
            B, H, W, C = x.shape
            if self.pos_embed is not None:
                pos_embed = resample_abs_pos_embed_3d(
                    self.pos_embed,
                    (H, W),
                    num_prefix_tokens=self.num_prefix_tokens,
                )
            else:
                pos_embed = None
            x = x.view(B, -1, C)
            rot_pos_embed = self.rope.get_embed(shape=(H, W)) if self.rope is not None else None
        else:
            pos_embed = self.pos_embed
            rot_pos_embed = self.rope.get_embed() if self.rope is not None else None

        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        if pos_embed is not None:
            x = x + pos_embed

        if self.reg_token is not None:
            to_cat = []
            if self.cls_token is not None:
                to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))
            x = torch.cat(to_cat + [x], dim=1)

        x = self.pos_drop(x)

        # obtain shared rotary position embedding and apply patch dropout
        if self.patch_drop is not None:
            x, keep_indices = self.patch_drop(x)
            if rot_pos_embed is not None and keep_indices is not None:
                rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)
            return x, rot_pos_embed, keep_indices
        else:
            return x, rot_pos_embed, None

    def forward_features(self, x, attn_mask: Optional[torch.Tensor] = None):
        x, rot_pos_embed, keep_indices = self._pos_embed(x)
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, rope=rot_pos_embed, attn_mask=attn_mask)
            else:
                x = blk(x, rope=rot_pos_embed, attn_mask=attn_mask)
        x = self.norm(x)
        return x, keep_indices

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        x, keep_indices = self.forward_features(x, attn_mask)
        return x, keep_indices


class EvaEncFromEvaMAE(nn.Module):
    def __init__(
            self,
            input_channels: int,
            embed_dim: int,
            patch_embed_size: Tuple[int, ...],
            output_channels: int,
            encoder_eva_depth: int = 24,
            encoder_eva_numheads: int = 16,
            decoder_eva_depth: int = 24,
            decoder_eva_numheads: int = 16,
            input_shape: Tuple[int, ...] = None,
            decoder_norm=LayerNormNd,
            decoder_act=nn.GELU,
            num_register_tokens: int = 0,
            class_token: bool = False,
            use_rot_pos_emb: bool = True,
            use_abs_pos_emb: bool = True,
            mlp_ratio=4 * 2 / 3,
            drop_path_rate=0,
            drop_path_scale: bool = True,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            rope_impl=RotaryEmbeddingCat,
            rope_kwargs=None
    ):

        assert input_shape is not None
        assert len(input_shape) == 3, "Currently only 3D is supported"
        assert all([j % i == 0 for i, j in zip(patch_embed_size, input_shape)])

        super().__init__()

        # Patch embedding for encoder
        self.down_projection = PatchEmbed(patch_embed_size, input_channels, embed_dim)

        self.eva = Eva(
            embed_dim=embed_dim,
            depth=encoder_eva_depth,
            num_heads=encoder_eva_numheads,
            ref_feat_shape=tuple([i // ds for i, ds in zip(input_shape, patch_embed_size)]),
            num_reg_tokens=num_register_tokens,
            use_rot_pos_emb=use_rot_pos_emb,
            use_abs_pos_emb=use_abs_pos_emb,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            drop_path_scale=drop_path_scale,
            patch_drop_rate=patch_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs,
            class_token=class_token
        )

        self.class_token = class_token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=1e-6)

    def restore_full_sequence(self, x, keep_indices, num_patches):
        """
        Restore the full sequence by filling blanks with mask tokens and reordering.
        Optimized for GPU utilization.
        """
        B, num_kept, C = x.shape
        device = x.device

        # Create mask tokens for missing patches
        num_masked = num_patches - num_kept
        mask_tokens = self.mask_token.repeat(B, num_masked, 1)  # Shape: (B, num_masked, C)

        # Prepare an empty tensor for the restored sequence
        restored = torch.zeros(B, num_patches, C, device=device)

        # Create a flat indices tensor for assignment
        batch_indices = torch.arange(B, device=device).unsqueeze(-1)  # Shape: (B, 1)

        # Flatten keep_indices to use for scatter
        flat_indices = torch.arange(num_patches, device=device).repeat(B, 1)  # Shape: (B, num_patches)
        mask = torch.zeros_like(flat_indices, dtype=torch.bool, device=device)  # Mask for "kept" indices

        # Mark kept positions
        mask[batch_indices, keep_indices] = True

        # Assign kept positions
        restored[batch_indices, keep_indices] = x

        # Assign mask tokens to the remaining positions
        masked_positions = torch.where(~mask)
        restored[masked_positions] = mask_tokens.view(-1, C)
        return restored

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        # Encode patches
        x = self.down_projection(x)
        B, C, W, H, D = x.shape
        x = rearrange(x, 'b c w h d -> b (h w d) c')

        # Encode using EVA (internally applies masking with patch_drop_rate)
        encoded, keep_indices = self.eva(x, attn_mask)
        # Restore full sequence with mask tokens
        num_patches = W * H * D
        if self.class_token:
            restored_x = self.restore_full_sequence(encoded[:, 1:, :], keep_indices, num_patches)
        else:
            restored_x = self.restore_full_sequence(encoded, keep_indices, num_patches)

        # Reshape back to 3D
        x = rearrange(restored_x, 'b (h w d) c -> b c w h d', h=H, w=W, d=D)

        return x


class PretrainedEvaEncFromEvaMAE(EvaEncFromEvaMAE):
    def __init__(
            self,
            input_channels: int,
            embed_dim: int,
            patch_embed_size: Tuple[int, ...],
            output_channels: int,
            encoder_eva_depth: int = 24,
            encoder_eva_numheads: int = 16,
            decoder_eva_depth: int = 24,
            decoder_eva_numheads: int = 16,
            input_shape: Tuple[int, ...] = None,
            decoder_norm=LayerNormNd,
            decoder_act=nn.GELU,
            num_register_tokens: int = 0,
            class_token: bool = False,
            use_rot_pos_emb: bool = True,
            use_abs_pos_emb: bool = True,
            mlp_ratio=4 * 2 / 3,
            drop_path_rate=0,
            drop_path_scale: bool = True,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            rope_impl=RotaryEmbeddingCat,
            rope_kwargs=None,
            pretrained_weights_file: str = None
    ):
        super().__init__(
            input_channels=input_channels,
            embed_dim=embed_dim,
            patch_embed_size=patch_embed_size,
            output_channels=output_channels,
            encoder_eva_depth=encoder_eva_depth,
            encoder_eva_numheads=encoder_eva_numheads,
            decoder_eva_depth=decoder_eva_depth,
            decoder_eva_numheads=decoder_eva_numheads,
            input_shape=input_shape,
            decoder_norm=decoder_norm,
            decoder_act=decoder_act,
            num_register_tokens=num_register_tokens,
            class_token=class_token,
            use_rot_pos_emb=use_rot_pos_emb,
            use_abs_pos_emb=use_abs_pos_emb,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            drop_path_scale=drop_path_scale,
            patch_drop_rate=patch_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs
        )

        if pretrained_weights_file is not None:
            self.load_pretrained_weights(pretrained_weights_file)
        else:
            raise ValueError("pretrained_weights_file must be provided")

    def load_pretrained_weights(self, pretrained_weights_file, verbose=True):
        # Load weights
        if dist.is_initialized():
            saved_model = torch.load(pretrained_weights_file, map_location=torch.device('cuda', dist.get_rank()), weights_only=False)
        else:
            saved_model = torch.load(pretrained_weights_file, weights_only=False)

        pretrained_dict = saved_model['network_weights']

        # Initialize model state dictionary
        model_dict = self.state_dict()

        # Filter out unnecessary keys based on match_encoder_only flag
        skip_strings_in_pretrained = ['.seg_layers.']
        skip_strings_in_pretrained.append('.decoder.')
        skip_strings_in_pretrained.append('up_projection')
        # skip_strings_in_pretrained.append('down_projection.proj.weight')

        # verify that all but the segmentation layers have the same shape
        for key, _ in model_dict.items():
            if all([i not in key for i in skip_strings_in_pretrained]):
                assert key in pretrained_dict, \
                    f"Key {key} is missing in the pretrained model weights. The pretrained weights do not seem to be " \
                    f"compatible with your network."
                if key not in ['down_projection.proj.weight']:
                    assert model_dict[key].shape == pretrained_dict[key].shape, \
                        f"The shape of the parameters of key {key} is not the same. Pretrained model: " \
                        f"{pretrained_dict[key].shape}; your network: {model_dict[key].shape}. The pretrained model " \
                        f"does not seem to be compatible with your network."

        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict.keys() and all([i not in k for i in skip_strings_in_pretrained])}

        channel_num = model_dict['down_projection.proj.weight'].shape[1]
        pretrained_dict['down_projection.proj.weight'] = pretrained_dict['down_projection.proj.weight'].repeat(1, channel_num, 1, 1, 1)

        model_dict.update(pretrained_dict)

        print("################### Loading pretrained weights from file ", pretrained_weights_file, '###################')
        if verbose:
            print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
            for key, value in pretrained_dict.items():
                print(key, 'shape', value.shape)
            print("################### Done ###################")
        self.load_state_dict(model_dict)


def get_config_from_plans(plans_file_path):
    # see here https://git.dkfz.de/mic/personal/group4/culrich/nnunet_finetuning/-/blob/fine_tuning_rocket/nnunetv2/training/nnUNetTrainer/variants/pretraining/eva_pretrained.py?ref_type=heads#L167

    plans = load_json(plans_file_path)
    pretrained_config = plans['configurations']['3d_fullres']

    def _get_ref_feat_shape(patch_size, vit_patch_size):
        return [int(np.ceil(patch_size_d / vit_patch_size_d)) for patch_size_d, vit_patch_size_d in zip(patch_size, vit_patch_size)]

    config = {
        # "vit_patch_size": pretrained_config['vit_patch_size'],
        "patch_embed_size": pretrained_config['vit_patch_size'],
        "input_shape": pretrained_config['patch_size'],
        "embed_dim": pretrained_config['embed_dim'],
        "encoder_eva_depth": pretrained_config['encoder_eva_depth'],
        "encoder_eva_numheads": pretrained_config['encoder_eva_numheads'],
        # "ref_feat_shape": _get_ref_feat_shape(pretrained_config['patch_size'], pretrained_config['vit_patch_size'])

    }
    return config


def pretrained_eva_enc_from_pretrain_folder(pretrain_folder: str, fold: Union[int, str] = 'all', checkpoint: str = "final"):
    plans_file_path = pretrain_folder + "/plans.json"
    config = get_config_from_plans(plans_file_path)
    pretrained_weights_file = pretrain_folder + f"/fold_{fold}/checkpoint_{checkpoint}.pth"
    eva_enc = PretrainedEvaEncFromEvaMAE(
        input_channels=1,
        output_channels=1,
        pretrained_weights_file=pretrained_weights_file,
        **config
    )
    return eva_enc



if __name__ == '__main__':
    embed_dim = round(1.5 * 16*40)
    pretrained_weights_file = "/home/r403k/Data/papertrack25/test_data/checkpoints/vision/Eva/checkpoint_final.pth"
    plans_file_path = "/home/r403k/Data/papertrack25/test_data/checkpoints/vision/Eva/plans.json"
    config = get_config_from_plans(plans_file_path)
    eva_enc = PretrainedEvaEncFromEvaMAE(
        input_channels=1,
        output_channels=1,
        pretrained_weights_file=pretrained_weights_file,
        **config
    )
    # input is B C W H D
    out = eva_enc(torch.rand(1, 1, 192, 192, 192))
    # output is B N T (where T is the token dimension and N is the number of tokens (i.e. W/Patch_embed_size_w * H/Patch_embed_size_h * D/Patch_embed_size_d))
    print(out)

    # patch_embed = PatchEmbed(
    #         img_size=224,
    #         patch_size=16,
    #         in_chans=3,
    #         embed_dim=512,
    #         dynamic_img_pad=False,
    #     )
    # b = patch_embed(torch.rand(2, 3, 224, 224)) # -> torch.Size([2, 196, 512])