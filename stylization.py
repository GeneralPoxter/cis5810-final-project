"""
CAP-VSTNet code adapted from:
https://github.com/linfengWen98/CAP-VSTNet
"""

import argparse
import os
import torch
from PIL import Image
from torchvision import transforms
import torchvision.utils as utils
from capvst.utils.utils import img_resize


def load_rev_network(ckpoint="capvst/checkpoints/art_image.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reversible Network
    from capvst.models.RevResNet import RevResNet

    RevNetwork = RevResNet(
        nBlocks=[10, 10, 10],
        nStrides=[1, 2, 2],
        nChannels=[16, 64, 256],
        in_channel=3,
        mult=4,
        hidden_dim=64,
        sp_steps=1,
    )

    state_dict = torch.load(ckpoint, map_location=device)
    RevNetwork.load_state_dict(state_dict["state_dict"])
    RevNetwork = RevNetwork.to(device)
    RevNetwork.eval()

    return RevNetwork


def image_transfer(
    content_path,
    style_path,
    RevNetwork,
    out_dir="output",
    max_size=1280,
):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Transfer module
    from capvst.models.cWCT import cWCT

    cwct = cWCT()

    content = Image.open(content_path).convert("RGB")
    style = Image.open(style_path).convert("RGB")

    ori_csize = content.size

    content = img_resize(content, max_size, down_scale=RevNetwork.down_scale)
    style = img_resize(style, max_size, down_scale=RevNetwork.down_scale)

    content = transforms.ToTensor()(content).unsqueeze(0).to(RevNetwork.device)
    style = transforms.ToTensor()(style).unsqueeze(0).to(RevNetwork.device)

    # Stylization
    with torch.no_grad():
        # Forward inference
        z_c = RevNetwork(content, forward=True)
        z_s = RevNetwork(style, forward=True)

        # Transfer
        z_cs = cwct.transfer(z_c, z_s)

        # Backward inference
        stylized = RevNetwork(z_cs, forward=False)

    # save stylized
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cn = os.path.basename(content_path)
    sn = os.path.basename(style_path)
    file_name = "%s_%s.png" % (cn.split(".")[0], sn.split(".")[0])
    path = os.path.join(out_dir, file_name)

    stylized = transforms.Resize(
        (ori_csize[1], ori_csize[0]), interpolation=Image.BICUBIC
    )(stylized)
    grid = utils.make_grid(stylized.data, nrow=1, padding=0)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    out_img = Image.fromarray(ndarr)

    out_img.save(path, quality=100)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpoint", type=str, default="capvst/checkpoints/art_image.pt"
    )

    # data
    parser.add_argument("--content", type=str, default="capvst/data/content/01.jpg")
    parser.add_argument("--style", type=str, default="capvst/data/style/01.jpg")

    parser.add_argument("--out_dir", type=str, default="output")
    parser.add_argument("--max_size", type=int, default=1280)

    args = parser.parse_args()

    RevNetwork = load_rev_network(args.ckpoint)
    path = image_transfer(
        args.content, args.style, RevNetwork, args.out_dir, args.max_size
    )
    print("Save at %s" % path)
