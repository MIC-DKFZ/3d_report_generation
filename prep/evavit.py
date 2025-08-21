# This is from https://github.com/MIC-DKFZ/nnssl/blob/openneuro/src/nnssl/architectures/evaMAE_module.py
from typing import Tuple, Optional, Union
import numpy as np

import torch
from torch import nn
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from timm.layers import RotaryEmbeddingCat
from torch import nn
from batchgenerators.utilities.file_and_folder_operations import load_json

from dynamic_network_architectures.building_blocks.eva import Eva
from dynamic_network_architectures.building_blocks.patch_encode_decode import PatchEmbed, PatchDecode, LayerNormNd
from dynamic_network_architectures.initialization.weight_init import InitWeights_He


class EvaWithAttentionMask(Eva):
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


class EvaMAE(nn.Module):
    def __init__(self,
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
                 use_rot_pos_emb: bool = True,
                 use_abs_pos_emb: bool = True,
                 mlp_ratio=4 * 2 / 3,
                 drop_path_rate=0,
                 drop_path_scale: bool = True,
                 patch_drop_rate: float = 0.,
                 proj_drop_rate: float = 0.,
                 attn_drop_rate: float = 0.2,
                 rope_impl=RotaryEmbeddingCat,
                 rope_kwargs=None,
                 do_up_projection=True,
                 init_values=None,
                 scale_attn_inner=False):
        """
        Masked Autoencoder with EVA attention-based encoder and decoder.
        """
        assert input_shape is not None
        assert len(input_shape) == 3, "Currently only 3D is supported"
        assert all([j % i == 0 for i, j in zip(patch_embed_size, input_shape)])

        super().__init__()
        self.patch_embed_size = patch_embed_size
        self.embed_dim = embed_dim
        # Patch embedding for encoder
        self.down_projection = PatchEmbed(patch_embed_size, input_channels, embed_dim)

        # Encoder using EVA
        self.eva = EvaWithAttentionMask(
            embed_dim=embed_dim,
            depth=encoder_eva_depth,
            num_heads=encoder_eva_numheads,
            ref_feat_shape=tuple([i // ds for i, ds in zip(input_shape, patch_embed_size)]),
            num_reg_tokens=num_register_tokens,
            use_rot_pos_emb=use_rot_pos_emb,
            use_abs_pos_emb=use_abs_pos_emb,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            patch_drop_rate=patch_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs,
            init_values=init_values,
            scale_attn_inner=scale_attn_inner
        )

        # Patch embedding for decoder
        if do_up_projection:
            self.up_projection = PatchDecode(patch_embed_size, embed_dim, output_channels,
                                             norm=decoder_norm,
                                             activation=decoder_act)
        else:
            self.up_projection = nn.Identity()

        # Decoder using EVA
        if decoder_eva_depth > 0:
            self.decoder = Eva(
                embed_dim=embed_dim,
                depth=decoder_eva_depth, #eva_depth,
                num_heads=decoder_eva_numheads, #eva_numheads,
                ref_feat_shape=tuple([i // ds for i, ds in zip(input_shape, patch_embed_size)]),
                num_reg_tokens=num_register_tokens,
                use_rot_pos_emb=use_rot_pos_emb,
                use_abs_pos_emb=use_abs_pos_emb,
                mlp_ratio=mlp_ratio,
                drop_path_rate=drop_path_rate,
                patch_drop_rate=0, # No drop in the decoder
                proj_drop_rate=proj_drop_rate,
                attn_drop_rate=attn_drop_rate,
                rope_impl=rope_impl,
                rope_kwargs=rope_kwargs,
                init_values=init_values,
                scale_attn_inner=scale_attn_inner
            )
            self.use_decoder = True
        else:
            self.use_decoder = False
            # self.decoder = DecoderIdentity()

        self.use_decoder = False #TODO make this less ugly

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=1e-6)

        self.down_projection.apply(InitWeights_He(1e-2))

    def restore_full_sequence(self, x, keep_indices, num_patches):
        """
        Restore the full sequence by filling blanks with mask tokens and reordering.
        Optimized for GPU utilization.
        """
        if keep_indices is None:
            return x
        B, num_kept, C = x.shape
        device = x.device

        # Create mask tokens for missing patches
        num_masked = num_patches - num_kept
        mask_tokens = self.mask_token.repeat(B, num_masked, 1)

        # Prepare an empty tensor for the restored sequence
        restored = torch.zeros(B, num_patches, C, device=device)

        # Assign the kept patches and mask tokens in the correct positions
        for i in range(B):
            kept_pos = keep_indices[i]
            masked_pos = torch.tensor([j for j in range(num_patches) if j not in kept_pos], device=device)
            restored[i, kept_pos] = x[i]
            restored[i, masked_pos] = mask_tokens[i, :len(masked_pos)]

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
        if self.use_decoder:
            restored_x = self.restore_full_sequence(encoded, keep_indices, num_patches)

            # Decode with restored sequence and rope embeddings
            decoded, _ = self.decoder(restored_x)
        else:
            decoded = encoded

        # Project back to output shape
        decoded = rearrange(decoded, 'b (h w d) c -> b c w h d', h=W, w=H, d=D)
        #decoded = self.up_projection(decoded)
        # THIS IS CUSTOM
        #here we directly return
        return decoded


class PretrainedEvaEncFromEvaMAE(EvaMAE):
    def __init__(
            self,
            input_channels: int,
            embed_dim: int,
            patch_embed_size: Tuple[int, ...],
            output_channels: int,
            encoder_eva_depth: int = 16,
            encoder_eva_numheads: int = 12,
            decoder_eva_depth: int = 8,
            decoder_eva_numheads: int = 12,
            input_shape: Tuple[int, ...] = None,
            decoder_norm=LayerNormNd,
            decoder_act=nn.GELU,
            num_register_tokens: int = 0,
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