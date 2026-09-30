from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
import torch
from nnunetv2.training.nnUNetTrainer.pretraining.resenc_example import (
    load_pretrained_weights,
)
import pydoc

class CustomUNetDecoder(UNetDecoder):
    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        stage_outputs = [lres_input]
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            stage_outputs.append(x)
            lres_input = x

        return stage_outputs

class ResidualEncoderUNetEncoderOnly(ResidualEncoderUNet):
    def __init__(self, **resenc_kwargs):
        super().__init__(**resenc_kwargs)
        self.decoder = CustomUNetDecoder(
            self.encoder,
            resenc_kwargs["num_classes"],
            resenc_kwargs["n_conv_per_stage_decoder"],
            resenc_kwargs.get("deep_supervision", False),
        )


def setup_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    arch = checkpoint["nnssl_adaptation_plan"]["architecture_plans"]["arch_kwargs"]
    requires_import = checkpoint["nnssl_adaptation_plan"]["architecture_plans"][
        "arch_kwargs_requiring_import"
    ]
    pre_train_statedict = checkpoint["network_weights"]
    config = checkpoint["nnssl_adaptation_plan"]

    for ri in requires_import:
        if arch[ri] is not None:
            arch[ri] = pydoc.locate(arch[ri])
    # Initializing Primus with the extracted architecture parameters
    resenc_kwargs = dict(
        input_channels=1,
        n_stages=arch["n_stages"],
        features_per_stage=arch["features_per_stage"],
        conv_op=arch["conv_op"],
        kernel_sizes=arch["kernel_sizes"],
        strides=arch["strides"],
        n_blocks_per_stage=arch["n_blocks_per_stage"],
        num_classes=1,
        n_conv_per_stage_decoder=arch["n_conv_per_stage_decoder"],
        conv_bias=arch["conv_bias"],
        norm_op=arch["norm_op"],
        norm_op_kwargs=arch["norm_op_kwargs"],
        nonlin=arch["nonlin"],
        nonlin_kwargs=arch["nonlin_kwargs"],
        deep_supervision=True,
    )
    model = ResidualEncoderUNetEncoderOnly(**resenc_kwargs)

    # this function is quite complicated, but allows for loading different checkpoints, stems, adapting positional embedding, skipping cls token...
    # we only load the "encoder" of the MAE. The MAE Decoder weights are skipped and up_projection conv layers stay random initialized
    model, _ = load_pretrained_weights(
        network=model,
        pre_train_statedict=pre_train_statedict,
        pt_input_channels=checkpoint["nnssl_adaptation_plan"][
            "pretrain_num_input_channels"
        ],
        downstream_input_channels=1,
        pt_input_patchsize=checkpoint["nnssl_adaptation_plan"][
            "recommended_downstream_patchsize"
        ],
        downstream_input_patchsize=config["recommended_downstream_patchsize"],
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
    checkpoint_path = "/home/s037r/Documents/mic_rocket_nnunete_finetuning/checkpoints/Dataset805_Rocket_v4/BaseMAETrainer_BS8_ep2500_ps192_Arch_Width_L_Depth_L_bs96_lr_5e2__nnsslPlans__noresample/fold_all/checkpoint_final.pth"
    model, _ = setup_model(checkpoint_path)
    x = torch.rand([1, 1, 192, 192, 192], device="cpu", dtype=torch.float32)

    with torch.no_grad():
        model.eval()
        output = model(x)
        for o in output:
            print(o.shape)
