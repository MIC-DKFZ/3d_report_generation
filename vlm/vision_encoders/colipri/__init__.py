import torch
from einops import rearrange
from .architecture_loading import build_global_embedding_network
import torch.nn as nn
import json
from pathlib import Path

class Colipri(nn.Module):
    def __init__(self, ckpt_path, adaptation_plan_path):
        super().__init__()
        self.model = build_global_embedding_network(
            ckpt_path=ckpt_path,
            adaptation_plan_path=adaptation_plan_path,
        )

    def forward(self, x):
        network, projection_layer, pooling_layer = self.model
        network_output = network(x)
        B, whd, d = network_output.shape
        w, h, d = 3 * [round(whd ** (1 / 3))]
        # [b, c_vision, w, h ,d] => [1, 864, 20, 20, 20]
        embeddings = rearrange(network_output, "b (w h d) c -> b c w h d", w=w, h=h, d=d)
        # [b, whd, c_language]  => [1, 8000, 768]
        proj_output = projection_layer(network_output)
        # [b, c_language]  => [1, 768]
        pooling_output = pooling_layer(proj_output)

        return embeddings, pooling_output


def setup_model(checkpoint_path):
    adaptation_plan_path = Path(checkpoint_path).parent.parent / "adaptation_plan.json"
    model = Colipri(
        ckpt_path=checkpoint_path,
        adaptation_plan_path=str(adaptation_plan_path),
    )

    config = json.loads(adaptation_plan_path.read_text())

    return model, config


if __name__ == "__main__":
    checkpoint_path = "/home/s037r/Documents/colibri/checkpoints/CLIPMAECAPTrainer_160_OS_vo1_MAEinit__nnsslPlans__onemmisopyramid/fold_all/checkpoint_final.pth"
    model, _ = setup_model(checkpoint_path)
    x = torch.rand([1, 1, 160, 160, 160], device="cpu", dtype=torch.float32).cuda()

    with torch.no_grad():
        model.eval()
        output = model(x)
        for o in output:
            print(o.shape)