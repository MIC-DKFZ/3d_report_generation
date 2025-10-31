from dynamic_network_architectures.architectures.primus import Primus
import torch
from einops import rearrange
from nnunetv2.training.nnUNetTrainer.pretraining.primus_example import (
    load_pretrained_weights,
)


class PrimusEncoderOnly(Primus):
    def forward(self, x, ret_mask=False):
        x = self.down_projection(x)
        # last output of the encoder is the input to EVA
        _, _, W, H, D = x.shape
        num_patches = W * H * D

        x = rearrange(x, "b c w h d -> b (w h d) c")
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x,
                ),
                dim=1,
            )
        x, keep_indices = self.eva(x)

        if self.register_tokens is not None:
            x = x[:, self.register_tokens.shape[1] :]  # Removes the register tokens
        # In-fill in-active patches with empty tokens
        restored_x, _ = self.restore_full_sequence(x, keep_indices, num_patches)
        x = rearrange(restored_x, "b (w h d) c -> b c w h d", h=H, w=W, d=D)
        return x


def setup_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    arch = checkpoint["nnssl_adaptation_plan"]["architecture_plans"]["arch_kwargs"]
    config = checkpoint["nnssl_adaptation_plan"]
    pre_train_statedict = checkpoint["network_weights"]

    primus_kwargs = dict(
        input_channels=1,
        embed_dim=arch["embed_dim"],
        patch_embed_size=tuple(arch["patch_embed_size"]),
        num_classes=1,
        eva_depth=arch["encoder_eva_depth"],  # !!! depth of the encoder
        eva_numheads=arch["encoder_eva_numheads"],  # !!! number of heads in the encoder
        input_shape=config["recommended_downstream_patchsize"],
        drop_path_rate=0.2,
        scale_attn_inner=True,
        init_values=0.1,
    )
    model = PrimusEncoderOnly(**primus_kwargs)

    # this function is quite complicated, but allows for loading different checkpoints, stems, adapting positional embedding, skipping cls token...
    # we only load the "encoder" of the MAE. The MAE Decoder weights are skipped and up_projection conv layers stay random initialized
    model, _ = load_pretrained_weights(
        network=model,
        pre_train_statedict=pre_train_statedict,
        pt_input_channels=checkpoint["nnssl_adaptation_plan"]["architecture_plans"][
            "arch_kwargs"
        ]["input_channels"],
        downstream_input_channels=1,
        pt_input_patchsize=checkpoint["nnssl_adaptation_plan"]["architecture_plans"][
            "arch_kwargs"
        ]["input_shape"],
        downstream_input_patchsize=checkpoint["nnssl_adaptation_plan"][
            "recommended_downstream_patchsize"
        ],
        pt_key_to_encoder=checkpoint["nnssl_adaptation_plan"]["key_to_encoder"],
        pt_key_to_stem=checkpoint["nnssl_adaptation_plan"]["key_to_stem"],
        pt_keys_to_in_proj=tuple(
            checkpoint["nnssl_adaptation_plan"]["keys_to_in_proj"]
        ),
        pt_key_to_lpe=checkpoint["nnssl_adaptation_plan"]["key_to_lpe"],
    )

    model.eval()
    return model, config


if __name__ == "__main__":
    checkpoint_path = "/home/s037r/Documents/mic_rocket_nnunete_finetuning/checkpoints/Dataset805_Rocket_v4/BaseEvaMAETrainer_BS96_192ps_2500ep_40_16_8_16_1056_lr2e3__nnsslPlans__noresample/fold_all/checkpoint_final.pth"
    model, _ = setup_model(checkpoint_path)
    x = torch.rand([1, 1, 192, 192, 192], device="cpu", dtype=torch.float32)

    with torch.no_grad():
        model.eval()
        output = model(x)
